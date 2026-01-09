from database import QDrantDB
import json
db = QDrantDB()
db.create_collection()
with open('data/chunks/chunks_with_id.json', 'r') as f:
    points = json.load(f)
db.upsert_points(points)