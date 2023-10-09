alert_control  = `<div class="control-group">

                    <label class="label-control" for="select-loc">Nivel de alerta actual:</label>
                    <div class="alert-panel-checkbox">
                        <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" id="check-fews" checked>
                        <label class="form-check-label" for="check-fews">Alertas FEWS</label>
                        </div>
                    </div>
                    <br>

                    <label class="label-control" for="select-loc">Niveles de alerta:</label>
                    <div class="alert-panel-checkbox">

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-lower-7" checked>
                            <label class="form-check-label" for="check-lower-7">Posibilidad de periodo seco alto</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-lower-3" checked>
                            <label class="form-check-label" for="check-lower-3">Posbilidad de periodo seco media</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-lower-1" checked>
                            <label class="form-check-label" for="check-lower-1">Posibilidad de periodo seco baja</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-002yr" checked>
                            <label class="form-check-label" for="check-002yr">Periodo de retorno: 2 años</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-005yr" checked>
                            <label class="form-check-label" for="check-005yr">Periodo de retorno: 5 años</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-010yr" checked>
                            <label class="form-check-label" for="check-010yr">Periodo de retorno: 10 años</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-025yr" checked>
                            <label class="form-check-label" for="check-025yr">Periodo de retorno: 25 años</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-050yr" checked>
                            <label class="form-check-label" for="check-050yr">Periodo de retorno: 50 años</label>
                        </div>

                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="check-100yr" checked>
                            <label class="form-check-label" for="check-100yr">Periodo de retorno: 100 años</label>
                        </div>

                    </div>
                    <br>
                </div>`;


function dynamic_select_alert(){
    
    $('#check-002yr').on('change', function () {
        if($('#check-002yr').is(':checked')){
            est_R002.addTo(map);
        } else {
            map.removeLayer(est_R002); 
        };
    });
    
    $('#check-005yr').on('change', function () {
        if($('#check-005yr').is(':checked')){
            est_R005.addTo(map);
        } else {
            map.removeLayer(est_R005); 
        };
    });
    
    $('#check-010yr').on('change', function () {
        if($('#check-010yr').is(':checked')){
            est_R010.addTo(map);
        } else {
            map.removeLayer(est_R010); 
        };
    });
    
    $('#check-025yr').on('change', function () {
        if($('#check-025yr').is(':checked')){
            est_R025.addTo(map);
        } else {
            map.removeLayer(est_R025); 
        };
    });
    
    $('#check-050yr').on('change', function () {
        if($('#check-050yr').is(':checked')){
            est_R050.addTo(map);
        } else {
            map.removeLayer(est_R050); 
        };
    });
    
    $('#check-100yr').on('change', function () {
        if($('#check-100yr').is(':checked')){
            est_R100.addTo(map);
        } else {
            map.removeLayer(est_R100); 
        };
    });
    
    $('#check-lower-1').on('change', function () {
        if($('#check-lower-1').is(':checked')){
            est_L_1.addTo(map);
        } else {
            map.removeLayer(est_L_1); 
        };
    });
    
    $('#check-lower-3').on('change', function () {
        if($('#check-lower-3').is(':checked')){
            est_L_3.addTo(map);
        } else {
            map.removeLayer(est_L_3); 
        };
    });
    
    $('#check-lower-7').on('change', function () {
        if($('#check-lower-7').is(':checked')){
            est_L_7.addTo(map);
        } else {
            map.removeLayer(est_L_7); 
        };
    });

    $('#check-fews').on('change', function () {
        if($('#check-fews').is(':checked')){
            fewsStaR0.addTo(map);
            fewsStaBass.addTo(map);
            fewsStaYl.addTo(map);
            fewsStaOr.addTo(map);
            fewsStaRd.addTo(map);
        } else {
            map.removeLayer(fewsStaR0);
            map.removeLayer(fewsStaBass);
            map.removeLayer(fewsStaYl);
            map.removeLayer(fewsStaOr);
            map.removeLayer(fewsStaRd);
        };
    });

};