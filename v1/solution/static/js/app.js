document.addEventListener('DOMContentLoaded', (event) => {
    const map = L.map('map', {
        center: [51.505, -0.09],
        zoom: 1,
        minZoom: 2,
        maxBounds: [
            [-90, -180],
            [90, 180]
        ],
        maxBoundsViscosity: 1.0
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
    }).addTo(map);

    const markers = {};
    let userId = null;
    let queryMode = false;
    let queriedElements = {};
    const drawnItems = new L.FeatureGroup(); // Track drawn circles

    map.addLayer(drawnItems);


    // Define custom icons
    const icons = {
        airplane: L.icon({
            iconUrl: '/static/img/airplane.png',
            iconSize: [32, 32], // size of the icon
        }),
        motorcycle: L.icon({
            iconUrl: '/static/img/motorcycle.png',
            iconSize: [32, 32],
        }),
        bike: L.icon({
            iconUrl: '/static/img/bike.png',
            iconSize: [32, 32],
        }),
        bird: L.icon({
            iconUrl: '/static/img/bird.png',
            iconSize: [32, 32],
        }),
        person: L.icon({
            iconUrl: '/static/img/person.png',
            iconSize: [32, 32],
        }),
    };

    const drawControl = new L.Control.Draw({
        draw: {
            polygon: true,
            polyline: true,
            rectangle: true,
            circle: true,
            marker: false
        },
        edit: {
            featureGroup: drawnItems // Assign feature group for editing
        }
    });
    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, function (e) {
        const type = e.layerType,
            layer = e.layer;

        if (type === 'circle') {
            const center = layer.getLatLng();
            const radius = layer.getRadius();

            // Perform query based on circle
            queryElementsInCircle(center, radius);
        } else if (type === 'polygon') {
            const points = layer.getLatLngs()[0].map(latlng => [latlng.lng, latlng.lat]);

            // Perform query based on polygon
            queryElementsInPolygon(points);
        }

        drawnItems.addLayer(layer); // Add the drawn layer to the feature group
    });



    document.getElementById('register-button2').addEventListener('click', () => {
        userId = document.getElementById('user-id').value;
        if (userId) {
            registerUser2(userId);
        }
    });

    document.getElementById('set-elements-button').addEventListener('click', () => {
        const numElements = parseInt(document.getElementById('num-elements').value);
        if (userId && !isNaN(numElements)) {
            registerUser2(userId, numElements);
        }
    });
    document.getElementById('set-speed-button').addEventListener('click', () => {
        const speed = parseFloat(document.getElementById('speed-control').value);
        if (!isNaN(speed) && speed > 0) {
            const socket = io();
            socket.emit('set_speed', { speed: speed });
            console.log(`Set movement speed to ${speed} seconds`);
        }
    });
    function registerUser2(userId, numElements = 10) {
        fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_id: userId, num_elements: numElements }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'registered') {
                    fetch(`/map_data/${userId}`)
                        .then(response => response.json())
                        .then(data => {
                            updateMarkers(data.elements);
                            setupWebSocket();
                        });
                }
            });
    }

    document.getElementById('query-type').addEventListener('change', (event) => {
        const queryType = event.target.value;
        document.getElementById('query-type-control').style.display = queryType === 'region' ? 'none' : 'block';
        document.getElementById('query-region-control').style.display = queryType === 'type' ? 'none' : 'block';
    });

    document.getElementById('query-button').addEventListener('click', () => {
        const queryType = document.getElementById('query-type').value;
        const elementType = document.getElementById('element-type').value;
        const userId = document.getElementById('user-id').value;
        const region = {
            lat_min: 0,
            lat_max: 0,
            lng_min: 0,
            lng_max: 0,
        };

        const queryData = {
            query_type: queryType,
            element_type: elementType,
            region: region,
            user_id: userId
        };

        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(queryData),
        })
            .then(response => response.json())
            .then(data => {
                clearMarkers();  // Clear existing markers
                queryMode = true; // Set query mode to true
                queriedElements = data.elements; // Store queried elements
                updateMarkers(data.elements);
            });
    });

    document.getElementById('show-all-button').addEventListener('click', () => {
        if (userId) {
            fetch(`/map_data/${userId}`)
                .then(response => response.json())
                .then(data => {
                    clearMarkers();  // Clear existing markers
                    queryMode = false; // Reset query mode
                    queriedElements = {}; // Clear queried elements
                    updateMarkers(data.elements);  // Show all elements
                });
        }
    });

    function queryElementsInCircle(center, radius) {
        const userId = document.getElementById('user-id').value;
        const queryData = {
            query_type: 'circle',
            center: center,
            radius: radius,
            user_id: userId
        };

        fetch('/query_circle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(queryData),
        })
            .then(response => response.json())
            .then(data => {
                clearMarkers();  // Clear existing markers
                clearCircles();  // Clear drawn circles
                queryMode = true; // Set query mode to true
                queriedElements = data.elements; // Store queried elements
                updateMarkers(data.elements);
            });
    }

    function queryElementsInPolygon(points) {
        const userId = document.getElementById('user-id').value;
        const queryData = {
            points: points,
            user_id: userId
        };

        fetch('/query_polygon', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(queryData),
        })
            .then(response => response.json())
            .then(data => {
                clearMarkers();  // Clear existing markers
                queryMode = true; // Set query mode to true
                queriedElements = data.elements; // Store queried elements
                updateMarkers(data.elements);
            });
    }
    function setupWebSocket() {
        const socket = io();
        socket.on('connect', () => {
            socket.emit('subscribe', { user_id: userId });
        });

        socket.on('update', (data) => {
            const parsedData = {
                id: data.id,
                type: data.type,
                lat: parseFloat(data.lat),
                lng: parseFloat(data.lng),
            };
            if (!queryMode || queriedElements[parsedData.id]) { // Update markers if not in query mode or if the element is part of the query
                updateMarkers({ [parsedData.id]: parsedData });
                if (queryMode) {
                    console.log('Query Mode received:', parsedData);
                    queriedElements[parsedData.id] = parsedData; // Ensure queried elements are updated
                }
                console.log('Update received:', parsedData);  // Debug print
            }
        });
    }

    function clearMarkers() {
        for (const id in markers) {
            map.removeLayer(markers[id]);
        }
        Object.keys(markers).forEach(id => delete markers[id]);
    }

    function clearCircles() {
        drawnItems.clearLayers(); // Clear all drawn circles
    }

    function updateMarkers(elements) {
        for (const [id, el] of Object.entries(elements)) {
            if (markers[id]) {
                markers[id].setLatLng([el.lat, el.lng]);
                markers[id].getPopup().setContent(`${el.id} - ${el.type} - ${el.lat.toFixed(4)}, ${el.lng.toFixed(4)}`);
            } else {
                const marker = L.marker([el.lat, el.lng], { icon: icons[el.type] }).addTo(map);
                marker.bindPopup(`${el.id} - ${el.type} - ${el.lat.toFixed(4)}, ${el.lng.toFixed(4)}`);
                markers[id] = marker;

                marker.on('click', () => {
                    marker.openPopup();
                });
            }
        }
    }
});
