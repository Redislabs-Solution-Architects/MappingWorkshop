from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from threading import Thread, Event
import math
import redis
import json
import random
import time
import threading
import pprint
import redis.commands.search
import redis.commands.search.aggregation as aggregations
import redis.commands.search.reducers as reducers
from redis.commands.json.path import Path
from redis.commands.search import Search
from redis.commands.search.field import (
    GeoField,
    GeoShapeField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import GeoFilter, NumericFilter, Query
from redis.commands.search.result import Result
from redis.commands.search.suggestion import Suggestion

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
r = redis.Redis(decode_responses=True)
r.flushdb()
ELEMENT_TYPES = ['airplane', 'motorcycle', 'bike', 'bird', 'person']
TOTAL_ELEMENTS = 10000
movement_speed = 2  # default speed in seconds


def create_index():
    schema = \
        (
            TextField("$.id", as_name="id"),
            TagField("$.type", as_name="type"),
            NumericField("$.lat", as_name="lat"),
            NumericField("$.lng", as_name="lng"),
            GeoField("$.location", as_name="location"),
            GeoShapeField("$.polylocation", as_name="poly_loc",
                          coord_system=GeoShapeField.SPHERICAL)
        )
    try:
        r.ft("elements_idx").create_index(schema, definition=IndexDefinition(
            prefix=["element:"], index_type=IndexType.JSON))
    except:
        print("error in index creation")


def generate_elements():
    for i in range(TOTAL_ELEMENTS):
        element_id = f"{i}"
        element_type = random.choice(ELEMENT_TYPES)
        lat = random.uniform(-90, 90)
        lng = random.uniform(-180, 180)
        speed = random.uniform(0.01, 0.1)  # Random speed
        direction = random.uniform(0, 360)  # Random direction in degrees
        element = {
            'id': element_id,
            'type': element_type,
            'speed': speed,
            'direction': direction,
            'lat': lat,
            'lng': lng,
            'location': f"{lng},{lat}",
            'polylocation': f"POINT ({lng} {lat})"
        }
        r.json().set(f"element:{element_id}", '.', element)
    print('Generated elements:', TOTAL_ELEMENTS)  # Debug print


def update_element_location3(element):
    direction = math.radians(element['direction'])
    speed = element['speed']
    delta_lat = speed * math.cos(direction)
    delta_lng = speed * math.sin(direction)

    element['lat'] += delta_lat
    element['lng'] += delta_lng

    # Keep the element within bounds
    element['lat'] = max(min(element['lat'], 85), -85)
    element['lng'] = max(min(element['lng'], 170), -170)
    element['location'] = f"{element['lng']},{element['lat']}"
    element['polylocation'] = f"POINT ({element['lng']} {element['lat']})"

    # Randomly change direction slightly to simulate natural movement
    element['direction'] += random.uniform(-10, 10)
    if element['direction'] >= 360 or element['direction'] < 0:
        element['direction'] = element['direction'] % 360

    return element


def update_element_location(element):
    print(f"before: {element}")
    direction = math.radians(element['direction'])
    speed = element['speed']
    delta_lat = speed * math.cos(direction)
    delta_lng = speed * math.sin(direction)

    if element['type'] == 'airplane':
        element['lat'] += delta_lat
        element['lng'] += delta_lng
    elif element['type'] == 'motorcycle':
        element['lat'] += delta_lat * 0.5
        element['lng'] += delta_lng * 0.5
    elif element['type'] == 'bike':
        element['lat'] += delta_lat * 0.3
        element['lng'] += delta_lng * 0.3
    elif element['type'] == 'bird':
        element['lat'] += delta_lat * 0.7
        element['lng'] += delta_lng * 0.7
    elif element['type'] == 'person':
        element['lat'] += delta_lat * 0.1
        element['lng'] += delta_lng * 0.1

    # Keep the element within bounds
    element['lat'] = max(min(element['lat'], 90), -90)
    element['lng'] = max(min(element['lng'], 180), -180)
    element['location'] = f"{element['lng']},{element['lat']}"
    element['location'] = f"{element['lng']},{element['lat']}"
    element['polylocation'] = f"POINT ({element['lng']} {element['lat']})"
    # Randomly change direction slightly to simulate natural movement
    element['direction'] += random.uniform(-10, 10)
    if element['direction'] >= 360 or element['direction'] < 0:
        element['direction'] = element['direction'] % 360
    print(f"after: {element}")
    return element


def update_element_location2(element, factor=50):
    if element['type'] == 'airplane':
        element['lat'] += random.uniform(-0.05, 0.05) * factor
        element['lng'] += random.uniform(-0.1, 0.1) * factor
    elif element['type'] == 'motorcycle':
        element['lat'] += random.uniform(-0.01, 0.01) * factor
        element['lng'] += random.uniform(-0.02, 0.02) * factor
    elif element['type'] == 'bike':
        element['lat'] += random.uniform(-0.005, 0.005) * factor
        element['lng'] += random.uniform(-0.01, 0.01) * factor
    elif element['type'] == 'bird':
        element['lat'] += random.uniform(-0.02, 0.02) * factor
        element['lng'] += random.uniform(-0.04, 0.04) * factor
    elif element['type'] == 'person':
        element['lat'] += random.uniform(-0.001, 0.001) * factor
        element['lng'] += random.uniform(-0.002, 0.002) * factor
    element['lat'] = max(min(element['lat'], 90), -90)
    element['lng'] = max(min(element['lng'], 180), -180)
    element['location'] = f"{element['lng']},{element['lat']}"
    element['polylocation'] = f"POINT ({element['lng']} {element['lat']})"
    return element


def update_elements():
    while True:
        for user_id in r.smembers('users'):
            user_id = user_id
            user_elements_key = f"user:{user_id}:elements"
            user_stream_key = f"user:{user_id}:stream"
            user_elements = r.smembers(user_elements_key)
            for element_id in user_elements:
                element = r.json().get(f"{element_id}")
                updated_element = update_element_location3(element)
                r.json().set(f"{element_id}", '.', updated_element)
                r.xadd(user_stream_key, updated_element)
        time.sleep(movement_speed)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    user_id = request.json['user_id']
    num_elements = request.json.get('num_elements', 10)
    r.sadd('users', user_id)
    user_elements_key = f"user:{user_id}:elements"
    user_stream_key = f"user:{user_id}:stream"
    elements_ids = random.sample(range(1, num_elements*2), num_elements)
    for key in elements_ids:
        r.sadd(user_elements_key, f'element:{key}')
        element = r.json().get(f'element:{key}')
        r.xadd(user_stream_key, element)
    print(f'Registered user {user_id} with elements:',
          elements_ids)  # Debug print
    return jsonify({'status': 'registered', 'user_id': user_id})


@app.route('/query', methods=['POST'])
def query_elements():
    data = request.json
    query_type = data.get('query_type')
    element_type = data.get('element_type')
    user_id = data.get('user_id')
    # Expecting region to be a dictionary with 'lat_min', 'lat_max', 'lng_min', 'lng_max'
    region = data.get('region')

    elements_res = {}
    if query_type == 'type':
        query = f"@type:{{{element_type}}}"
    elif query_type == 'region':
        lat_min = region['lat_min']
        lat_max = region['lat_max']
        lng_min = region['lng_min']
        lng_max = region['lng_max']
        query = f"@location:[{lng_min} {lat_min} {lng_max} {lat_max}]"
    elif query_type == 'type_region':
        lat_min = region['lat_min']
        lat_max = region['lat_max']
        lng_min = region['lng_min']
        lng_max = region['lng_max']
        query = f"@type:{element_type} @location:[{lng_min} {lat_min} {lng_max} {lat_max}]"
    else:
        query = "*"
    q = Query(query).paging(0, 10000)
    result = r.ft("elements_idx").search(q)
    for doc in result.docs:
        element_id = doc.id
        if r.sismember(f'user:{user_id}:elements', element_id):
            element = json.loads(doc.json)
            element_type = element['type']
            lat = element['lat']
            lng = element['lng']
            elements_res[element_id] = {'id': element_id,
                                        'type': element_type, 'lat': lat, 'lng': lng}

    return jsonify({"elements": elements_res})


@app.route('/query_polygon', methods=['POST'])
def query_polygon():
    data = request.json
    points = data['points']  # Expecting a list of [lng, lat] pairs
    user_id = data.get('user_id')
    # Convert points to a string format suitable for Redis
    polygon = " ".join([f"{p[0]} {p[1]}," for p in points])

    elements_res = {}
    query = f"@poly_loc:[within $shape]"
    query_params = {"shape": f"POLYGON(({polygon}))"}
    q = Query(query,).dialect(3).paging(
        0, 10000)
    result = r.ft("elements_idx").search(
        query=q, query_params=query_params)
    for doc in result.docs:
        element_id = doc.id
        if r.sismember(f'user:{user_id}:elements', element_id):
            element = json.loads(doc.json)[0]
            element_type = element['type']
            lat = element['lat']
            lng = element['lng']
            elements_res[element_id] = {'id': element_id,
                                        'type': element_type, 'lat': lat, 'lng': lng}

    return jsonify({"elements": elements_res})


@app.route('/query_circle', methods=['POST'])
def query_circle():
    data = request.json
    center = data['center']
    radius = data['radius'] / 1000
    user_id = data['user_id']
    elements_res = {}
    query = f"@location:[{center['lng']} {center['lat']} {radius} km]"
    q = Query(query).paging(0, 10000)
    result = r.ft("elements_idx").search(query=q)
    for doc in result.docs:
        element_id = doc.id
        if r.sismember(f'user:{user_id}:elements', element_id):
            element = json.loads(doc.json)
            element_type = element['type']
            lat = element['lat']
            lng = element['lng']
            elements_res[element_id] = {'id': element_id,
                                        'type': element_type, 'lat': lat, 'lng': lng}

    return jsonify({"elements": elements_res})


@socketio.on('set_speed')
def handle_set_speed(data):
    global movement_speed
    movement_speed = data['speed']
    print(f'Set movement speed to {movement_speed} seconds')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('subscribe')
def handle_subscribe(data):
    user_id = data['user_id']
    stream_key = f"user:{user_id}:stream"
    last_id_key = f"user:{user_id}:last_id"

    last_id = r.get(last_id_key) or '0'
    print(f'Subscribed to {stream_key} starting from {last_id}')

    while True:
        messages = r.xread(
            streams={stream_key: last_id}, count=100, block=1000)
        if messages:
            for message in messages[0][1]:
                last_id = message[0]
                msg = {k: v for k, v in message[1].items()}
                msg['id'] = f"element:{msg.get('id')}"
                emit('update', msg)
                print(f'Pushed update to {user_id}:', msg)  # Debug print
                r.set(last_id_key, last_id)
        else:
            # Sleep for a short time before trying again if no messages
            time.sleep(1)


@socketio.on('set_speed')
def handle_set_speed(data):
    global movement_speed
    movement_speed = data['speed']
    print(f'Set movement speed to {movement_speed} seconds')


@app.route('/map_data/<user_id>')
def get_map_data(user_id):
    user_elements_key = f"user:{user_id}:elements"
    elements = {}
    for element_id in r.smembers(user_elements_key):
        element = r.json().get(f"{element_id}")
        elements[element_id] = element
    return jsonify({"elements": elements})


create_index()
generate_elements()
update_thread = threading.Thread(target=update_elements)
update_thread.start()

if __name__ == '__main__':
    print('Starting application...')
    create_index()
    generate_elements()
    update_thread = Thread(target=update_elements)
    update_thread.daemon = True
    update_thread.start()
    socketio.run(app, host='0.0.0.0', port=8000)
