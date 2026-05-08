from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
import re
import networkx as nx
from collections import defaultdict

# ── Load model ────────────────────────────────────────────
MODEL_PATH = Path(r'G:\financial_kg\models\financial_ner_finbert_v2')
tokenizer  = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model      = AutoModelForTokenClassification.from_pretrained(str(MODEL_PATH))
model.eval()

with open(MODEL_PATH / 'label_map.json') as f:
    label_data = json.load(f)
ID2LABEL = {int(k): v for k, v in label_data['id2label'].items()}

# ── Relation patterns ─────────────────────────────────────
RELATION_PATTERNS = [
    # ACQUIRED
    (r'(\w[\w\s]+?)\s+acquired\s+(\w[\w\s]+?)(?:\s+for|\s*[,.])',
     'ACQUIRED'),
    (r'(\w[\w\s]+?)\s+bought\s+(\w[\w\s]+?)(?:\s+for|\s*[,.])',
     'ACQUIRED'),
    (r'(\w[\w\s]+?)\s+purchased\s+(\w[\w\s]+?)(?:\s+for|\s*[,.])',
     'ACQUIRED'),
    # CEO_OF
    (r'(\w[\w\s]+?)\s+CEO\s+(\w[\w\s]+)',
     'CEO_OF'),
    (r'(\w[\w\s]+?),?\s+(?:chief executive|ceo)\s+of\s+(\w[\w\s]+)',
     'CEO_OF'),
    # REPORTED
    (r'(\w[\w\s]+?)\s+reported\s+(?:revenue|income|profit|earnings|net income)\s+of\s+(\$[\d.,]+\s*(?:billion|million|trillion)?)',
     'REPORTED_REVENUE'),
    (r'(\w[\w\s]+?)\s+reported\s+(\$[\d.,]+\s*(?:billion|million|trillion)?)',
     'REPORTED_REVENUE'),
    # FINED
    (r'(\w[\w\s]+?)\s+fined\s+(\$[\d.,]+\s*(?:billion|million|trillion)?)',
     'FINED'),
    (r'(\w[\w\s]+?)\s+paid\s+(\$[\d.,]+\s*(?:billion|million|trillion)?)\s+(?:fine|penalty|settlement)',
     'FINED'),
    # LOCATED_IN
    (r'(\w[\w\s]+?)\s+(?:headquartered|based|located)\s+in\s+(\w[\w\s]+?)(?:\s+reported|\s*[,.])',
     'LOCATED_IN'),
    # INVESTED_IN
    (r'(\w[\w\s]+?)\s+invested\s+(\$[\d.,]+\s*(?:billion|million|trillion)?)\s+in\s+(\w[\w\s]+)',
     'INVESTED_IN'),
    # PARTNERED
    (r'(\w[\w\s]+?)\s+(?:partnered|partnership)\s+with\s+(\w[\w\s]+)',
     'PARTNERED_WITH'),
    # SURGED/FELL
    (r'(\w[\w\s]+?)\s+shares?\s+(?:surged|jumped|rose|climbed)',
     'STOCK_SURGED'),
    (r'(\w[\w\s]+?)\s+shares?\s+(?:fell|dropped|declined|slumped)',
     'STOCK_FELL'),
]

app = FastAPI(
    title='Financial NER + Knowledge Graph API',
    version='2.0.0',
    description='Financial NER with Knowledge Graph — FinBERT + NetworkX'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class ExtractionRequest(BaseModel):
    text: str


class EntitySpan(BaseModel):
    text      : str
    label     : str
    start     : int
    end       : int
    confidence: float = 0.0


class GraphNode(BaseModel):
    id    : str
    label : str
    type  : str
    size  : float = 1.0


class GraphEdge(BaseModel):
    source  : str
    target  : str
    relation: str
    weight  : float = 1.0


class ExtractionResponse(BaseModel):
    text        : str
    entities    : list[EntitySpan]
    entity_count: int


class GraphResponse(BaseModel):
    text    : str
    entities: list[EntitySpan]
    nodes   : list[GraphNode]
    edges   : list[GraphEdge]
    stats   : dict


def extract_entities(text: str) -> list[EntitySpan]:
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop('offset_mapping')[0].tolist()

    with torch.no_grad():
        outputs = model(**inputs)

    logits      = outputs.logits[0]
    probs       = torch.softmax(logits, dim=-1)
    pred_ids    = torch.argmax(probs, dim=-1).tolist()
    confidences = torch.max(probs, dim=-1).values.tolist()
    tokens      = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    word_preds = []
    for token, pred_id, conf, offset in zip(tokens, pred_ids, confidences, offset_mapping):
        if offset[0] == 0 and offset[1] == 0:
            continue
        is_subword = token.startswith('##')
        if is_subword and word_preds:
            word_preds[-1]['end']  = offset[1]
            word_preds[-1]['text'] = text[word_preds[-1]['start']:offset[1]]
        else:
            word_preds.append({
                'text'      : text[offset[0]:offset[1]],
                'label'     : ID2LABEL.get(pred_id, 'O'),
                'start'     : offset[0],
                'end'       : offset[1],
                'confidence': round(conf * 100, 1),
            })

    entities       = []
    current_entity = None

    for wp in word_preds:
        label = wp['label']
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'text'       : wp['text'],
                'label'      : label[2:],
                'start'      : wp['start'],
                'end'        : wp['end'],
                'confidence' : wp['confidence'],
                'conf_sum'   : wp['confidence'],
                'conf_count' : 1,
            }
        elif label.startswith('I-') and current_entity:
            current_entity['text']        = text[current_entity['start']:wp['end']]
            current_entity['end']         = wp['end']
            current_entity['conf_sum']   += wp['confidence']
            current_entity['conf_count'] += 1
            current_entity['confidence']  = round(
                current_entity['conf_sum'] / current_entity['conf_count'], 1
            )
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Fix money spans
    MONEY_EXTENSIONS = ['billion', 'million', 'trillion', 'thousand']
    processed = []
    for entity in entities:
        if entity['label'] == 'MONEY':
            end       = entity['end']
            remaining = text[end:end+20]
            match = re.match(
                r'^[.,]\d+\s*(billion|million|trillion|thousand|mn|bn)?',
                remaining, re.I
            )
            if match:
                entity['end']  = end + len(match.group(0).rstrip())
                entity['text'] = text[entity['start']:entity['end']]
            else:
                match2 = re.match(
                    r'^\s*(billion|million|trillion|thousand|mn|bn)',
                    remaining, re.I
                )
                if match2:
                    entity['end']  = end + len(match2.group(0).rstrip())
                    entity['text'] = text[entity['start']:entity['end']]
        processed.append(entity)

    return [EntitySpan(
        text       = e['text'],
        label      = e['label'],
        start      = e['start'],
        end        = e['end'],
        confidence = e['confidence'],
    ) for e in processed]


def extract_relations(text: str, entities: list[EntitySpan]) -> list[tuple]:
    relations = []
    text_lower = text.lower()

    for pattern, relation in RELATION_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if len(groups) == 2:
                subj = groups[0].strip()
                obj  = groups[1].strip()
                # Verify entities exist in our NER output
                subj_entity = next(
                    (e for e in entities if e.text.lower() in subj.lower()
                     or subj.lower() in e.text.lower()), None
                )
                obj_entity = next(
                    (e for e in entities if e.text.lower() in obj.lower()
                     or obj.lower() in e.text.lower()), None
                )
                if subj_entity and obj_entity and subj_entity.text != obj_entity.text:
                    relations.append((
                        subj_entity.text,
                        subj_entity.label,
                        relation,
                        obj_entity.text,
                        obj_entity.label,
                    ))

    return relations


def build_graph(entities: list[EntitySpan], relations: list[tuple]) -> tuple:
    G = nx.DiGraph()

    # Add entity nodes
    entity_counts = defaultdict(int)
    for entity in entities:
        entity_counts[entity.text] += 1

    for entity in entities:
        if not G.has_node(entity.text):
            G.add_node(
                entity.text,
                type=entity.label,
                size=entity_counts[entity.text],
            )

    # Add relation edges
    for subj, subj_type, relation, obj, obj_type in relations:
        if not G.has_node(subj):
            G.add_node(subj, type=subj_type, size=1)
        if not G.has_node(obj):
            G.add_node(obj, type=obj_type, size=1)
        G.add_edge(subj, obj, relation=relation, weight=1.0)

    nodes = [
        GraphNode(
            id    = node,
            label = node,
            type  = data.get('type', 'UNKNOWN'),
            size  = float(data.get('size', 1)),
        )
        for node, data in G.nodes(data=True)
    ]

    edges = [
        GraphEdge(
            source   = u,
            target   = v,
            relation = data.get('relation', 'RELATED_TO'),
            weight   = data.get('weight', 1.0),
        )
        for u, v, data in G.edges(data=True)
    ]

    stats = {
        'nodes'      : G.number_of_nodes(),
        'edges'      : G.number_of_edges(),
        'density'    : round(nx.density(G), 4),
        'components' : nx.number_weakly_connected_components(G),
    }

    return nodes, edges, stats


# ── Routes ────────────────────────────────────────────────
@app.get('/health')
def health():
    return {
        'status'  : 'ok',
        'model'   : 'FinBERT NER + NetworkX KG',
        'version' : '2.0.0',
        'entities': ['PER', 'ORG', 'LOC', 'MONEY', 'DATE'],
    }


@app.post('/extract', response_model=ExtractionResponse)
def extract(request: ExtractionRequest):
    entities = extract_entities(request.text)
    return ExtractionResponse(
        text         = request.text,
        entities     = entities,
        entity_count = len(entities),
    )


@app.post('/graph', response_model=GraphResponse)
def build_knowledge_graph(request: ExtractionRequest):
    entities  = extract_entities(request.text)
    relations = extract_relations(request.text, entities)
    nodes, edges, stats = build_graph(entities, relations)

    return GraphResponse(
        text     = request.text,
        entities = entities,
        nodes    = nodes,
        edges    = edges,
        stats    = stats,
    )