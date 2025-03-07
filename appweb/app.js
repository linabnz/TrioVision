// Créer la carte OpenLayers
const map = new ol.Map({
    target: 'map', // ID du conteneur HTML pour la carte
    layers: [
        new ol.layer.Tile({
            source: new ol.source.OSM() // Utilise OpenStreetMap comme source
        })
    ],
    view: new ol.View({
        center: ol.proj.fromLonLat([0, 20]), // Centre initial [longitude, latitude]
        zoom: 2 // Niveau de zoom
    })
});

// Gestion de l'upload d'image
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const predictedCountry = document.getElementById('predicted-country');
const probabilitiesTable = document.getElementById('probabilities-table');

let imageUploaded = false; // Variable pour savoir si une image a été uploadée

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Afficher l'image uploadée
            dropZone.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image" style="max-width: 100%; border-radius: 10px;">`;
            imageUploaded = true;
            // Simuler une prédiction (à remplacer par une requête API)
            simulatePrediction();
        };
        reader.readAsDataURL(file);
    }
});

// Charger le GeoJSON du monde et ajouter une couche vectorielle
const vectorSource = new ol.source.Vector({
    url: 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json',
    format: new ol.format.GeoJSON()
});

const vectorLayer = new ol.layer.Vector({
    source: vectorSource,
    style: function (feature) {
        if (!imageUploaded) {
            return new ol.style.Style({
                fill: new ol.style.Fill({ color: 'rgba(200, 200, 200, 0.5)' }), // Gris par défaut
                stroke: new ol.style.Stroke({ color: '#000', width: 1 })
            });
        }
        
        const countryName = feature.get('name');
        const normalizedCountry = normalizeCountryName(countryName);
        if (normalizedCountry === bestCountry) {
            return new ol.style.Style({
                fill: new ol.style.Fill({ color: 'rgba(0, 0, 255, 0.7)' }), // Bleu marqué uniquement pour le pays avec la plus haute probabilité
                stroke: new ol.style.Stroke({ color: '#000', width: 1 })
            });
        }
        return new ol.style.Style({
            fill: new ol.style.Fill({ color: 'rgba(200, 200, 200, 0.5)' }), // Les autres pays restent gris
            stroke: new ol.style.Stroke({ color: '#000', width: 1 })
        });
    }
});

// Ajouter la couche vectorielle à la carte
map.addLayer(vectorLayer);

// Ajouter un marqueur pour afficher la localisation du pays avec la plus grande probabilité
const markerSource = new ol.source.Vector();
const markerLayer = new ol.layer.Vector({
    source: markerSource
});
map.addLayer(markerLayer);

function addMarker(lon, lat) {
    markerSource.clear(); 
    const markerFeature = new ol.Feature({
        geometry: new ol.geom.Point(ol.proj.fromLonLat([lon, lat]))
    });
    markerFeature.setStyle(new ol.style.Style({
        image: new ol.style.Circle({
            radius: 8,
            fill: new ol.style.Fill({ color: 'red' }),
            stroke: new ol.style.Stroke({ color: 'white', width: 2 })
        })
    }));
    markerSource.addFeature(markerFeature);
}

function normalizeCountryName(name) {
    return name;
}


let probabilitiesMap = {};
let bestCountry = '';


function updateMapWithProbabilities(probabilities, countries) {
    probabilitiesMap = {};
    let highestProbability = 0;
    
    let tableHTML = '<table border="1" style="width:100%; text-align:left;"><tr><th>Pays</th><th>Probabilité</th></tr>';
    
    countries.forEach((country, index) => {
        probabilitiesMap[country] = probabilities[index];
        tableHTML += `<tr><td>${country}</td><td>${(probabilities[index] * 100).toFixed(2)}%</td></tr>`;
        
        if (probabilities[index] > highestProbability) {
            highestProbability = probabilities[index];
            bestCountry = country;
        }
    });
    
    tableHTML += '</table>';
    if (probabilitiesTable) {
        probabilitiesTable.innerHTML = tableHTML;
    }
    
    console.log('Mise à jour des probabilités:', probabilitiesMap); // Debug
    vectorLayer.getSource().changed();
    
    
    if (bestCountry) {
        fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${bestCountry}`)
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    const lon = parseFloat(data[0].lon);
                    const lat = parseFloat(data[0].lat);
                    addMarker(lon, lat);
                }
            });
    }
}


function simulatePrediction() {
    const countries = ['France', 'USA', 'Japon', 'Brésil', 'Australie'];
    const probabilities = [Math.random(), Math.random(), Math.random(), Math.random(), Math.random()]; // Générer des probabilités aléatoires
    
    
    const maxProb = Math.max(...probabilities);
    bestCountry = countries[probabilities.indexOf(maxProb)];
    
   
    predictedCountry.innerText = `Pays prédit : ${bestCountry}`;
    
    
    updateMapWithProbabilities(probabilities, countries);
}
