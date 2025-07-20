document.addEventListener('DOMContentLoaded', (event) => {
    // Map initialization
    const map = L.map('map', {
        center: [51.505, -0.09],
        zoom: 2,
        minZoom: 1,
        maxBounds: [[-90, -180], [90, 180]],
        maxBoundsViscosity: 1.0
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Global variables
    const markers = {};
    let userId = null;
    let queryMode = false;
    let queriedElements = {};
    let selectedElement = null;
    let socket = null;
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    // Custom icons with enhanced styling
    const icons = {
        airplane: L.icon({
            iconUrl: '/static/img/airplane.png',
            iconSize: [28, 28],
            className: 'marker-airplane'
        }),
        motorcycle: L.icon({
            iconUrl: '/static/img/motorcycle.png',
            iconSize: [28, 28],
            className: 'marker-motorcycle'
        }),
        bike: L.icon({
            iconUrl: '/static/img/bike.png',
            iconSize: [28, 28],
            className: 'marker-bike'
        }),
        bird: L.icon({
            iconUrl: '/static/img/bird.png',
            iconSize: [28, 28],
            className: 'marker-bird'
        }),
        person: L.icon({
            iconUrl: '/static/img/person.png',
            iconSize: [28, 28],
            className: 'marker-person'
        }),
    };

    // Enhanced drawing controls
    const drawControl = new L.Control.Draw({
        draw: {
            polygon: true,
            polyline: false,
            rectangle: true,
            circle: true,
            marker: false
        },
        edit: {
            featureGroup: drawnItems
        }
    });
    map.addControl(drawControl);

    // Drawing event handlers
    map.on(L.Draw.Event.CREATED, function (e) {
        const type = e.layerType;
        const layer = e.layer;

        if (type === 'circle') {
            const center = layer.getLatLng();
            const radius = layer.getRadius();
            queryElementsInCircle(center, radius);
        } else if (type === 'polygon') {
            const points = layer.getLatLngs()[0].map(latlng => [latlng.lng, latlng.lat]);
            queryElementsInPolygon(points);
        } else if (type === 'rectangle') {
            const bounds = layer.getBounds();
            const region = {
                lat_min: bounds.getSouth(),
                lat_max: bounds.getNorth(),
                lng_min: bounds.getWest(),
                lng_max: bounds.getEast()
            };
            queryElementsByRegion(region);
        }

        drawnItems.addLayer(layer);
    });

    // UI Event Listeners
    setupEventListeners();
    setupImageUpload();
    setupWebSocket();
    function setupEventListeners() {
        // User registration
        document.getElementById('register-button').addEventListener('click', () => {
            userId = document.getElementById('user-id').value.trim();
            if (userId) {
                registerUser(userId);
            } else {
                showAlert('Please enter a valid user ID', 'warning');
            }
        });

        // Add elements
        document.getElementById('add-elements-button').addEventListener('click', () => {
            const numElements = parseInt(document.getElementById('num-elements').value);
            if (userId && !isNaN(numElements) && numElements > 0) {
                registerUser(userId, numElements);
            } else {
                showAlert('Please enter a valid number of elements', 'warning');
            }
        });

        // Speed control
        const speedSlider = document.getElementById('speed-slider');
        const speedDisplay = document.getElementById('speed-display');
        
        speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            speedDisplay.textContent = speed + 's';
            if (socket) {
                socket.emit('set_speed', { speed: speed });
            }
        });

        // Traditional query
        document.getElementById('query-button').addEventListener('click', executeTraditionalQuery);

        // Show all elements
        document.getElementById('show-all-button').addEventListener('click', showAllElements);

        // Clear search
        document.getElementById('clear-search-button').addEventListener('click', clearSearch);

        // Natural language search
        document.getElementById('naturalSearchBtn').addEventListener('click', executeNaturalLanguageSearch);

        // Enter key for natural language input
        document.getElementById('naturalLanguageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                executeNaturalLanguageSearch();
            }
        });

        // Find similar elements
        document.getElementById('findSimilarBtn').addEventListener('click', findSimilarElements);

        // Anomaly detection
        document.getElementById('anomalyBtn').addEventListener('click', detectAnomalies);

        // Anomaly threshold slider
        const anomalySlider = document.getElementById('anomaly-threshold');
        const thresholdDisplay = document.getElementById('threshold-display');
        
        anomalySlider.addEventListener('input', (e) => {
            thresholdDisplay.textContent = e.target.value;
        });

        // Image search
        document.getElementById('imageSearchBtn').addEventListener('click', executeImageSearch);

        // Query type change
        document.getElementById('query-type').addEventListener('change', (event) => {
            const queryType = event.target.value;
            const typeControl = document.getElementById('query-type-control');
            typeControl.style.display = queryType === 'region' ? 'none' : 'block';
        });
        
    }

    function startVoice() {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
      
        recognition.onresult = function(event) {
          const query = event.results[0][0].transcript;
          console.log("You said:", query);
      
          fetch('/enhanced_natural_search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query_text: query,
              user_id: 'your-user-id'
            })
          })
          .then(response => response.json())
          .then(data => {
            console.log("Search result:", data);
            // update UI here
          });
        };
      
        recognition.onerror = function(e) {
          console.error('Voice recognition error:', e);
        };
      
        recognition.start();
    }

    function setupImageUpload() {
        const uploadArea = document.getElementById('imageUploadArea');
        const fileInput = document.getElementById('imageInput');
        const searchBtn = document.getElementById('imageSearchBtn');

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // File selection
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleImageFile(file);
                searchBtn.disabled = false;
                uploadArea.innerHTML = `
                    <i class="fas fa-check-circle fa-2x mb-2 text-success"></i>
                    <p class="mb-0">Image: ${file.name}</p>
                    <small class="text-success">Ready to search</small>
                `;
            }
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                fileInput.files = files;
                handleImageFile(files[0]);
                searchBtn.disabled = false;
                uploadArea.innerHTML = `
                    <i class="fas fa-check-circle fa-2x mb-2 text-success"></i>
                    <p class="mb-0">Image: ${files[0].name}</p>
                    <small class="text-success">Ready to search</small>
                `;
            }
        });
    }

    function handleImageFile(file) {
        // Store the file for later use
        window.selectedImageFile = file;
        
        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            console.log('Image loaded for search:', file.name);
        };
        reader.readAsDataURL(file);
    }

    function setupWebSocket() {
        socket = io();
        
        socket.on('connect', () => {
            updateConnectionStatus(true);
            console.log('Connected to server');
            if (userId) {
                socket.emit('subscribe', { user_id: userId });
            }
        });

        socket.on('disconnect', () => {
            updateConnectionStatus(false);
            console.log('Disconnected from server');
        });

        socket.on('update', (data) => {
            // Parse any JSON strings back to objects/arrays
            const parsedData = parseStreamData(data);
            
            const elementData = {
                id: parsedData.id,
                type: parsedData.type,
                lat: parseFloat(parsedData.lat),
                lng: parseFloat(parsedData.lng),
            };
            
            // Update markers if not in query mode or if element is part of query
            if (!queryMode || queriedElements[elementData.id]) {
                updateMarkers({ [elementData.id]: elementData });
                if (queryMode) {
                    queriedElements[elementData.id] = elementData;
                }
            }
        });

        socket.on('anomaly_alert', (data) => {
            showAnomalyAlert(data);
        });
    }

    function updateConnectionStatus(connected) {
        const statusIndicator = document.getElementById('connectionStatus');
        if (connected) {
            statusIndicator.className = 'status-indicator status-online';
            statusIndicator.title = 'Connected to Redis';
        } else {
            statusIndicator.className = 'status-indicator status-offline';
            statusIndicator.title = 'Disconnected';
        }
    }

    function registerUser(userId, numElements = 10) {
        showLoading(true);
        
        fetch('/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, num_elements: numElements }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'registered') {
                showAlert(`Registered user ${userId} with ${numElements} elements`, 'success');
                loadUserElements(userId);
                if (socket) {
                    socket.emit('subscribe', { user_id: userId });
                }
            }
        })
        .catch(error => {
            console.error('Registration error:', error);
            showAlert('Registration failed', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
    }

    function loadUserElements(userId) {
        fetch(`/map_data/${userId}`)
            .then(response => response.json())
            .then(data => {
                clearMarkers();
                updateMarkers(data.elements);
                queryMode = false;
                queriedElements = {};
            })
            .catch(error => {
                console.error('Error loading elements:', error);
                showAlert('Failed to load elements', 'error');
            });
    }

    function executeTraditionalQuery() {
        if (!userId) {
            showAlert('Please register first', 'warning');
            return;
        }

        const queryType = document.getElementById('query-type').value;
        const elementType = document.getElementById('element-type').value;
        
        const queryData = {
            query_type: queryType,
            element_type: elementType,
            region: { lat_min: 0, lat_max: 0, lng_min: 0, lng_max: 0 },
            user_id: userId
        };

        showLoading(true);
        
        fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryData),
        })
        .then(response => response.json())
        .then(data => {
            displaySearchResults(data.elements, 'Traditional Query');
            clearMarkers();
            queryMode = true;
            queriedElements = data.elements;
            updateMarkers(data.elements);
        })
        .catch(error => {
            console.error('Query error:', error);
            showAlert('Query failed', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
    }

    function executeNaturalLanguageSearch() {
        const queryText = document.getElementById('naturalLanguageInput').value.trim();
        
        if (!queryText) {
            showAlert('Please enter a search query', 'warning');
            return;
        }

        if (!userId) {
            showAlert('Please register first', 'warning');
            return;
        }

        showLoading(true);
        
        fetch('/natural_language_search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query_text: queryText,
                user_id: userId,
                k: 50
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'error');
                return;
            }
            
            displaySearchResults(data.elements, `"${queryText}"`);
            clearMarkers();
            queryMode = true;
            queriedElements = data.elements;
            updateMarkers(data.elements);
            
            showAlert(`Found ${Object.keys(data.elements).length} elements`, 'success');
        })
        .catch(error => {
            console.error('Natural language search error:', error);
            showAlert('Search failed', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
    }

    function executeImageSearch() {
        if (!window.selectedImageFile) {
            showAlert('Please select an image first', 'warning');
            return;
        }

        if (!userId) {
            showAlert('Please register first', 'warning');
            return;
        }

        const formData = new FormData();
        formData.append('image', window.selectedImageFile);
        formData.append('user_id', userId);
        formData.append('k', '30');

        showLoading(true);
        
        fetch('/image_search', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'error');
                return;
            }
            
            displaySearchResults(data.elements, 'Image Search', true);
            clearMarkers();
            queryMode = true;
            queriedElements = data.elements;
            updateMarkers(data.elements);
            
            showAlert(`Found ${Object.keys(data.elements).length} visually similar elements`, 'success');
        })
        .catch(error => {
            console.error('Image search error:', error);
            showAlert('Image search failed', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
    }

    function findSimilarElements() {
        if (!selectedElement) {
            showAlert('Please click on an element first', 'warning');
            return;
        }

        if (!userId) {
            showAlert('Please register first', 'warning');
            return;
        }

        const similarityType = document.getElementById('similarity-type').value;
        
        showLoading(true);
        
        fetch('/find_similar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_element_id: selectedElement,
                user_id: userId,
                similarity_type: similarityType,
                k: 20
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'error');
                return;
            }
            
            displaySearchResults(data.elements, `Similar to ${selectedElement} (${similarityType})`, true);
            clearMarkers();
            queryMode = true;
            queriedElements = data.elements;
            updateMarkers(data.elements);
            
            // Highlight the original element
            if (markers[selectedElement]) {
                markers[selectedElement].setIcon(createHighlightIcon(markers[selectedElement].options.icon));
            }
            
            showAlert(`Found ${Object.keys(data.elements).length} similar elements`, 'success');
        })
        .catch(error => {
            console.error('Similarity search error:', error);
            showAlert('Similarity search failed', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
    }

    function detectAnomalies() {
        if (!userId) {
            showAlert('Please register first', 'warning');
            return;
        }

        const threshold = parseFloat(document.getElementById('anomaly-threshold').value);
        
        showLoading(true);
        
        fetch('/anomaly_detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                threshold: threshold
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'error');
                return;
            }
            
            displayAnomalyResults(data.anomalous_elements);
            clearMarkers();
            queryMode = true;
            queriedElements = data.anomalous_elements;
            updateMarkers(data.anomalous_elements);
            
            const count = Object.keys(data.anomalous_elements).length;
            if (count > 0) {
                showAlert(`Detected ${count} anomalous elements`, 'warning');
            } else {
                showAlert('No anomalies detected', 'success');
            }
        })
        .catch(error => {
            console.error('Anomaly detection error:', error);
            showAlert('Anomaly detection failed', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
    }

    

    function queryElementsInCircle(center, radius) {
        if (!userId) return;
        
        const queryData = {
            center: center,
            radius: radius,
            user_id: userId
        };

        fetch('/query_circle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryData),
        })
        .then(response => response.json())
        .then(data => {
            displaySearchResults(data.elements, 'Circle Query');
            clearMarkers();
            queryMode = true;
            queriedElements = data.elements;
            updateMarkers(data.elements);
        })
        .catch(error => console.error('Circle query error:', error));
    }

    function queryElementsInPolygon(points) {
        if (!userId) return;
        
        const queryData = {
            points: points,
            user_id: userId
        };

        fetch('/query_polygon', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryData),
        })
        .then(response => response.json())
        .then(data => {
            displaySearchResults(data.elements, 'Polygon Query');
            clearMarkers();
            queryMode = true;
            queriedElements = data.elements;
            updateMarkers(data.elements);
        })
        .catch(error => console.error('Polygon query error:', error));
    }

    function queryElementsByRegion(region) {
        if (!userId) return;
        
        const queryData = {
            query_type: 'region',
            region: region,
            user_id: userId
        };

        fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryData),
        })
        .then(response => response.json())
        .then(data => {
            displaySearchResults(data.elements, 'Rectangle Query');
            clearMarkers();
            queryMode = true;
            queriedElements = data.elements;
            updateMarkers(data.elements);
        })
        .catch(error => console.error('Region query error:', error));
    }

    function showAllElements() {
        if (!userId) {
            showAlert('Please register first', 'warning');
            return;
        }
        
        loadUserElements(userId);
        clearSearchResults();
        clearDrawnItems();
    }

    function clearSearch() {
        clearMarkers();
        clearSearchResults();
        clearDrawnItems();
        clearAnomalyAlerts();
        queryMode = false;
        queriedElements = {};
        selectedElement = null;
        document.getElementById('findSimilarBtn').disabled = true;
        
        // Reset image upload
        const uploadArea = document.getElementById('imageUploadArea');
        uploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt fa-2x mb-2 text-muted"></i>
            <p class="mb-0">Drop image here or click to upload</p>
            <small class="text-muted">Find elements similar to your image</small>
        `;
        document.getElementById('imageSearchBtn').disabled = true;
        document.getElementById('imageInput').value = '';
        window.selectedImageFile = null;
        
        showAlert('Search cleared', 'info');
    }

    function clearMarkers() {
        Object.values(markers).forEach(marker => map.removeLayer(marker));
        Object.keys(markers).forEach(id => delete markers[id]);
    }

    function clearDrawnItems() {
        drawnItems.clearLayers();
    }

    function clearSearchResults() {
        document.getElementById('searchResults').style.display = 'none';
        document.getElementById('resultsContent').innerHTML = '';
    }

    function clearAnomalyAlerts() {
        document.getElementById('anomalyAlerts').innerHTML = '';
    }

    function updateMarkers(elements) {
        Object.entries(elements).forEach(([id, el]) => {
            if (markers[id]) {
                // Update existing marker
                markers[id].setLatLng([el.lat, el.lng]);
                updateMarkerPopup(markers[id], el);
            } else {
                // Create new marker
                const marker = L.marker([el.lat, el.lng], { 
                    icon: icons[el.type] 
                }).addTo(map);
                
                updateMarkerPopup(marker, el);
                markers[id] = marker;

                // Enhanced marker click handler
                marker.on('click', () => {
                    selectedElement = id;
                    document.getElementById('findSimilarBtn').disabled = false;
                    marker.openPopup();
                    
                    // Highlight selected marker
                    Object.values(markers).forEach(m => {
                        if (m !== marker) {
                            m.setIcon(icons[m.elementType]);
                        }
                    });
                    marker.setIcon(createHighlightIcon(icons[el.type]));
                    
                    showAlert(`Selected ${el.type} ${id}`, 'info');
                });
            }
        });
    }

    function updateMarkerPopup(marker, element) {
        const similarityScore = element.similarity_score ? 
            `<br><strong>Similarity:</strong> ${(element.similarity_score * 100).toFixed(1)}%` : '';
        
        const anomalyScore = element.anomaly_score ? 
            `<br><strong>Anomaly Score:</strong> ${(element.anomaly_score * 100).toFixed(1)}%` : '';
        
        const anomalyReasons = element.anomaly_reasons ? 
            `<br><strong>Issues:</strong> ${element.anomaly_reasons.join(', ')}` : '';
        
        marker.bindPopup(`
            <strong>${element.id}</strong><br>
            <strong>Type:</strong> ${element.type}<br>
            <strong>Location:</strong> ${element.lat.toFixed(4)}, ${element.lng.toFixed(4)}
            ${similarityScore}
            ${anomalyScore}
            ${anomalyReasons}
        `);
        
        marker.elementType = element.type;
    }

    function createHighlightIcon(baseIcon) {
        // Create a highlighted version of the icon
        const highlightIcon = L.icon({
            iconUrl: baseIcon.options.iconUrl,
            iconSize: [36, 36], // Slightly larger
            className: 'marker-highlighted'
        });
        return highlightIcon;
    }

    function displaySearchResults(elements, searchType, showScores = false) {
        const resultsContainer = document.getElementById('searchResults');
        const resultsContent = document.getElementById('resultsContent');
        
        resultsContainer.style.display = 'block';
        
        const elementArray = Object.entries(elements);
        if (elementArray.length === 0) {
            resultsContent.innerHTML = '<p class="text-muted">No results found</p>';
            return;
        }
        
        // Sort by similarity score if available
        if (showScores) {
            elementArray.sort((a, b) => (b[1].similarity_score || 0) - (a[1].similarity_score || 0));
        }
        
        let html = `<small class="text-muted">${searchType} - ${elementArray.length} results</small><br>`;
        
        elementArray.slice(0, 10).forEach(([id, element]) => {
            const scoreText = showScores && element.similarity_score ? 
                `<span class="similarity-score">${(element.similarity_score * 100).toFixed(1)}%</span>` : '';
            
            html += `
                <div class="result-item" onclick="focusOnElement('${id}')">
                    <strong>${element.type}</strong> ${id.replace('element:', '')} ${scoreText}<br>
                    <small class="text-muted">${element.lat.toFixed(3)}, ${element.lng.toFixed(3)}</small>
                </div>
            `;
        });
        
        if (elementArray.length > 10) {
            html += `<small class="text-muted">... and ${elementArray.length - 10} more</small>`;
        }
        
        resultsContent.innerHTML = html;
    }

    function displayAnomalyResults(anomalousElements) {
        const resultsContainer = document.getElementById('searchResults');
        const resultsContent = document.getElementById('resultsContent');
        
        resultsContainer.style.display = 'block';
        
        const elementArray = Object.entries(anomalousElements);
        if (elementArray.length === 0) {
            resultsContent.innerHTML = '<p class="text-success">No anomalies detected</p>';
            return;
        }
        
        // Sort by anomaly score
        elementArray.sort((a, b) => (b[1].anomaly_score || 0) - (a[1].anomaly_score || 0));
        
        let html = `<small class="text-warning">Anomaly Detection - ${elementArray.length} anomalies</small><br>`;
        
        elementArray.forEach(([id, element]) => {
            const scorePercent = (element.anomaly_score * 100).toFixed(1);
            html += `
                <div class="result-item" onclick="focusOnElement('${id}')">
                    <strong class="text-warning">${element.type}</strong> ${id.replace('element:', '')}<br>
                    <small>Score: <span class="text-danger">${scorePercent}%</span></small><br>
                    <small class="text-muted">${element.lat.toFixed(3)}, ${element.lng.toFixed(3)}</small>
                </div>
            `;
        });
        
        resultsContent.innerHTML = html;
    }

    function showAnomalyAlert(anomalyData) {
        const alertsContainer = document.getElementById('anomalyAlerts');
        const alertHtml = `
            <div class="anomaly-alert">
                <strong><i class="fas fa-exclamation-triangle"></i> Anomaly Detected!</strong><br>
                ${anomalyData.element_type} ${anomalyData.element_id}<br>
                <small>${anomalyData.reason}</small>
            </div>
        `;
        alertsContainer.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            const alert = alertsContainer.firstElementChild;
            if (alert) alert.remove();
        }, 10000);
    }

    // Global functions
    window.focusOnElement = function(elementId) {
        const marker = markers[elementId];
        if (marker) {
            map.setView(marker.getLatLng(), Math.max(map.getZoom(), 6));
            marker.openPopup();
            selectedElement = elementId;
            document.getElementById('findSimilarBtn').disabled = false;
        }
    };

    window.toggleSection = function(sectionId) {
        const section = document.getElementById(sectionId);
        const isVisible = section.style.display !== 'none';
        section.style.display = isVisible ? 'none' : 'block';
        
        // Rotate arrow icon
        const header = section.previousElementSibling;
        const arrow = header.querySelector('.fa-chevron-down');
        if (arrow) {
            arrow.style.transform = isVisible ? 'rotate(-90deg)' : 'rotate(0deg)';
        }
    };

    function showLoading(show) {
        document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
    }

    function showAlert(message, type = 'info') {
        // Create toast-like alert
        const alertClass = {
            'success': 'alert-success',
            'error': 'alert-danger', 
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';
        
        const alert = document.createElement('div');
        alert.className = `alert ${alertClass} alert-dismissible fade show`;
        alert.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            min-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        `;
        alert.innerHTML = `
            ${message}
            <button type="button" class="close" onclick="this.parentElement.remove()">
                <span>&times;</span>
            </button>
        `;
        
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }

    // Initialize connection status
    updateConnectionStatus(false);
    
    console.log('Enhanced Redis Vector Search Workshop initialized');

    // Helper function to parse stream data
    function parseStreamData(data) {
        const parsed = { ...data };
        
        // List of keys that might contain JSON strings
        const jsonKeys = ['visual_vector', 'behavior_vector', 'semantic_vector', 'path_history'];
        
        jsonKeys.forEach(key => {
            if (parsed[key] && typeof parsed[key] === 'string') {
                try {
                    parsed[key] = JSON.parse(parsed[key]);
                } catch (e) {
                    // Keep as string if parsing fails
                    console.warn(`Failed to parse ${key}:`, e);
                }
            }
        });
        
        return parsed;
    }
});