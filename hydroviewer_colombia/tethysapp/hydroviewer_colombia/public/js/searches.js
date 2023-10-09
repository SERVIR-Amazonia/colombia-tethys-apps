search_control = `
    <div class="control-group">
        <label class="label-control" for="select-fews-station">Estación hidrológica:</label>
        <select id="select-fews-station" multiple placeholder="Escriba el código o nombre de la estación."></select>
        <br></br>
    </div>
`;



function dynamic_search_boxes(){

    fetch("get-fews-alerts")
        .then((response) => (layer = response.json()))
        .then((layer) => {
            // Format json as input of selectize
            fews_layer = layer.features.map(item => item.properties);

            $("#select-fews-station").selectize({
                maxItems:     1,
                options:      fews_layer,
                valueField:  'nombre',
                labelField:  'nombre',
                searchField: ['nombre'],
                create : false,
                onChange : function(value, isOnInitialize) {
                    est_item = fews_layer.filter(item => item.nombre == value)[0];

                    if (typeof ss_marker !== 'undefined') {
                        map.removeLayer(ss_marker)
                    }

                    ss_marker = L.circleMarker([est_item.lat, est_item.lng], {
                        radius : 7,
                        color  : '#AD2745',
                        opacity: 0.75,
                    }).addTo(map);

                    // Bounds
                    southWest = L.latLng(est_item.lat - 0.01, est_item.lng - 0.01);
                    northEast = L.latLng(est_item.lat + 0.01, est_item.lng + 0.01);
                    bounds    = L.latLngBounds(southWest, northEast);

                    // Fit the map
                    map.fitBounds(bounds);
                },
            });
        });

};