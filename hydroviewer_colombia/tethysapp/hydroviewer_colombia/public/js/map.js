// ------------------------------------------------------------------------------------------------------------ //
//                                          GLOBAL AND STATE VARIABLES                                          //
// ------------------------------------------------------------------------------------------------------------ //

// Server domain (DNS or IP:port)
const server = "http://localhost:8080";




// ------------------------------------------------------------------------------------------------------------ //
//                                              INITIALIZE THE MAP                                              //
// ------------------------------------------------------------------------------------------------------------ //

// Ajust the map to the window height
const height = $(window).height() - 50;
$("#map-container").height(height);

// Set the map container
var map = L.map("map-container", {
    zoomControl: false,
}).setView([4.5988, -74.08], 5);

// Add the base map
L.tileLayer("http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

// Add the zoom control
L.control.zoom({ 
    position: "bottomright"
}).addTo(map);


// ------------------------------------------------------------------------------------------------------------ //
//                                     COLOR MARKER ACCORDING TO THE ALERT                                      //
// ------------------------------------------------------------------------------------------------------------ //

// Function to construct Icon Marker
function IconMarker(rp) {
  const IconMarkerR = new L.Icon({
    iconUrl: `${server}/static/hydroviewer_colombia/images/icon_popup/${rp}.svg`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
  return IconMarkerR;
}

// Function to construct Icon Marker
function IconMarkerF(c, size) {
    const IconMarkerR = new L.Icon({
      iconUrl: `${server}/static/hydroviewer_colombia/images/icon_popup/${c}.png`,
      iconSize: [size, size],
      iconAnchor: [size/2.0, size/2.0],
    });
    return IconMarkerR;
  }

// Icon markers for each return period
const IconR000 = IconMarker("0");       // RP: 0 years
const IconR002 = IconMarker("2");       // RP: 2 years
const IconR005 = IconMarker("5");       // RP: 5 years
const IconR010 = IconMarker("10");      // RP: 10 years
const IconR025 = IconMarker("25");      // RP: 25 years
const IconR050 = IconMarker("50");      // RP: 50 years
const IconR100 = IconMarker("100");     // RP: 100 years

const IconL_1 = IconMarker("lower_1");  // Low 7q10 from 1 to 3 days
const IconL_3 = IconMarker("lower_3");  // Low 7q10 from 3 to 7 days
const IconL_7 = IconMarker("lower_7");  // Low 7q10 from 7 to plus days

// Icon markers for each return period
const IconFR0 = IconMarkerF("normal", 10); 
const IconFB = IconMarkerF("bass", 18);
const IconFY = IconMarkerF("y", 14); 
const IconFO = IconMarkerF("o", 16);
const IconFR = IconMarkerF("r", 18); 


// Customized icon function
function IconParse(feature, latlng) {
    switch (feature.properties.alert) {
        case "R0":
            StationIcon = IconR000;
            break;
        case "R2":
            StationIcon = IconR002;
            break;
        case "R5":
            StationIcon = IconR005;
            break;
        case "R10":
            StationIcon = IconR010;
            break;
        case "R25":
            StationIcon = IconR025;
            break;
        case "R50":
            StationIcon = IconR050;
            break;
        case "R100":
            StationIcon = IconR100;
            break;

        case "lower_1":
            StationIcon = IconL_1;
            break;
        case "lower_3":
            StationIcon = IconL_3;
            break;
        case "lower_7":
            StationIcon = IconL_7;
            break;

    }
    return L.marker(latlng, { icon: StationIcon });
}

// Customized icon function for FEWS
function IconParseFEWS(feature, latlng) {
    switch(feature.properties.alert) {
        case "R0":
            StationIcon = IconFR0;
            break;
        case "-bass":
            StationIcon = IconFB;
            break;
        case "-O":
            StationIcon = IconFO;
            break;
        case "-R":
            StationIcon = IconFR;
            break;
        case "-Y":
            StationIcon = IconFY;
            break;
    }
    return L.marker(latlng, {icon : StationIcon})
}


function color_alert(alert) {
    switch(alert) {
        case "R0":
            color = "#fff0";
            break;
        case "-bass":
            color = "#bc1cd4";
            break;
        case "-O":
            color = "#d47e1c7d";
            break;
        case "-R":
            color = "#ff00157d";
            break;
        case "-Y":
            color = "#f1ff007d";
            break; 
    };
    return color
}

// ------------------------------------------------------------------------------------------------------------ //
//                                            PANEL DATA INFORMATION                                            //
// ------------------------------------------------------------------------------------------------------------ //
const sleep = ms => new Promise(r => setTimeout(r, ms));

var global_comid;
var global_station;

async function get_data_fews(code, name, lat, lon, depto, mun, cenpoblado, nomZH, nomSZH, alert,
    umaxhis, ubajos, uamarilla, unaranja, uroja) {
    
    // Updating the comid
    global_station = code;

    // Add data to the panel
    $("#station-code-custom-fw").html(`<b>CÓDIGO:</b> &nbsp ${code}`);
    $("#station-name-custom-fw").html(`<b>NOMBRE:</b> &nbsp ${name}`);
    $("#station-latitude-custom-fw").html(`<b>LATITUD:</b> &nbsp ${lat}`);
    $("#station-longitude-custom-fw").html(`<b>LONGITUD:</b> &nbsp ${lon}`);
    $("#station-depto-custom-fw").html(`<b>DEPARTAMENTO:</b> &nbsp ${depto}`);
    $("#station-municipio-custom-fw").html(`<b>MUNICIPIO:</b> &nbsp ${mun}`);
    $("#station-locality1-custom-fw").html(`<b>CENTRO POBLADO:</b> &nbsp ${cenpoblado}`);
    $("#station-locality2-custom-fw").html(`<b>ZONA HIDROGRÁFICA:</b> &nbsp ${nomZH}`);
    $("#station-locality3-custom-fw").html(`<b>SUBZONA HIDROGRÁFICA:</b> &nbsp ${nomSZH}`);

    $("#panel-modal-fews").css("border", `10px solid ${color_alert(alert)}`);

    loader = `<div class="loading-container" style="height: 350px; padding-top: 12px;"> 
                <div class="loading"> 
                  <h2>LOADING DATA</h2>
                  <span></span><span></span><span></span><span></span><span></span><span></span><span></span> 
                </div>
              </div>`; 
    
    // Add the dynamic loader
    $("#streamflow").html(loader);
    $("#water-level").html(loader);
    $("#streamflow-forecast").html(loader);
    $("#water-level-forecast").html(loader);

    // We need stop 300ms to obtain the width of the panel-tab-content-fews
    await sleep(300);

    $.ajax({
        type: 'GET', 
        url: "get-data-fews",
        data: {
            code: code,
            width: `${$("#panel-tab-content-fews").width()}`,
            umaxhis : umaxhis, 
            ubajos : ubajos, 
            uamarilla : uamarilla, 
            unaranja : unaranja, 
            uroja : uroja,
        }
    }).done(function(response){
        $("#modal-body-panel-custom-fews").html(response)
    });

};


async function get_data_station(comid, depto, mun, nomAH, nomZH, nomSZH){
    // Updating the comid
    global_comid = comid
    
    // Add data to the panel
    $("#station-comid-custom").html(`<b>COMID:</b> &nbsp ${comid}`);
    $("#station-depto-custom").html(`<b>DEPARTAMENTO: </b> &nbsp ${depto}`);
    $("#station-mun-custom").html(`<b>MUNICIPIO: </b> &nbsp ${mun}`);
    $("#station-locality1-custom").html(`<b>ÁREA HIDROGRÁFICA:</b> &nbsp ${nomAH}`);
    $("#station-locality2-custom").html(`<b>ZONA HIDROGRÁFICA :</b> &nbsp ${nomZH}`);
    $("#station-locality3-custom").html(`<b>SUBZONA HIDROGRÁFICA :</b> &nbsp ${nomSZH}`);

    loader = `<div class="loading-container" style="height: 350px; padding-top: 12px;"> 
                <div class="loading"> 
                  <h2>LOADING DATA</h2>
                  <span></span><span></span><span></span><span></span><span></span><span></span><span></span> 
                </div>
              </div>`; 

    // Add the dynamic loader
    $("#hydrograph").html(loader)
    $("#visual-analisis").html(loader)
    $("#metrics").html(loader)
    $("#forecast").html(loader)
    $("#corrected-forecast").html(loader)

    // We need stop 300ms to obtain the width of the panel-tab-content
    await sleep(300);

    // Retrieve the data
    $.ajax({
        type: 'GET', 
        url: "get-data",
        data: {
            comid: comid,
            width: `${$("#panel-tab-content").width()}`
        }
    }).done(function(response){
        $("#modal-body-panel-custom").html(response)
    })
}




// ------------------------------------------------------------------------------------------------------------ //
//                                         ADDING THE DRAINAGE NETWORK                                          //
// ------------------------------------------------------------------------------------------------------------ //

window.onload = function () {
  
    // Show data panel
    function showPanel_comid(comid, depto, mun, nomAH, nomZH, nomSZH) {
        $("#panel-modal").modal("show");
        // get_data_station(comid, name, lat, lon, loc1, loc2)
        get_data_station(comid, depto, mun, nomAH, nomZH, nomSZH);
    }


    function showPanelRiver(e) {
        var comid = e.target.feature.properties.comid;
        var depto = e.target.feature.properties.DPTO_CNMBR ;
        var mun = e.target.feature.properties.NOMBRE_MPI ;
        var nomAH = e.target.feature.properties.NOMAH ;
        var nomZH = e.target.feature.properties.NOMZH ;
        var nomSZH = e.target.feature.properties.NOMSZH ;

        showPanel_comid(comid, depto, mun, nomAH, nomZH, nomSZH);
    }


    function showPanelAlert(e) {
        var comid  = e.layer.feature.properties.comid;
        var depto  = e.layer.feature.properties.loc0;
        var mun = e.layer.feature.properties.loc3;
        var nomAH = e.layer.feature.properties.loc4;
        var nomZH  = e.layer.feature.properties.loc1;
        var nomSZH = e.layer.feature.properties.loc2;

        showPanel_comid(comid, depto, mun, nomAH, nomZH, nomSZH);
    }


    function showPanelFews(e) {
        // Load data for feratures
        var code = e.layer.feature.properties.id;
        var name = e.layer.feature.properties.nombre;
        var lat = e.layer.feature.properties.lat;
        var lon = e.layer.feature.properties.lng;
        var depto = e.layer.feature.properties.depart;
        var mun = e.layer.feature.properties.municipio;
        var cenpoblado = e.layer.feature.properties.cenpoblado;
        var nomZH = e.layer.feature.properties.zona;
        var nomSZH = e.layer.feature.properties.subzona;
        var alert = e.layer.feature.properties.alert;

        var umaxhis   = e.layer.feature.properties.umaxhis;
        var ubajos    = e.layer.feature.properties.ubajos;
        var uamarilla = e.layer.feature.properties.uamarilla;
        var unaranja  = e.layer.feature.properties.unaranja;
        var uroja     = e.layer.feature.properties.uroja;

        // Turn on panel modal fews
        $("#panel-modal-fews").modal("show");

        // Main
        get_data_fews(code, name, lat, lon, depto, mun, cenpoblado, nomZH, nomSZH, alert,
            umaxhis, ubajos, uamarilla, unaranja, uroja);
        
    }


    // Load drainage network 
    fetch("get-drainage-json", {
        method: 'GET',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response=>response.json())
    .then(resp => {
        console.log(resp);
        var drainage = new L.geoJson(resp,
            {onEachFeature : function(feature, layer){
                layer.on({
                    // On click function
                    click : function(e){
                        showPanelRiver(e);
                        },
                    mouseover: function(e) {
                        e.target.setStyle({
                                weight: 3, // Cambia el grosor del trazo
                                color: "#FF0000" // Cambia el color
                            });
                        },
                    mouseout: function(e) {
                        e.target.setStyle({
                                weight: 1, // Cambia el grosor del trazo
                                color: "#4747C9", // Cambia el color
                                opacity: 0.5
                            });
                        },
                });
                },
            style : {
                weight: 1, 
                color: "#4747C9",
                opacity: 0.5
                }
            }).addTo(map);
        
        // Fit the map to the river bounds
        map.fitBounds(drainage.getBounds());
    }).catch(error => console.error('Error fetching data:', error));

    // Load FEWS stations
    fetch("get-fews-alerts")
    .then((response) => (layer = response.json()))
    .then((layer) => {

        fewsStaR0 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R0"), {
            pointToLayer : IconParseFEWS,
        });
        fewsStaR0.addTo(map);
        fewsStaR0.on('click', showPanelFews);

        fewsStaBass = L.geoJSON(layer.features.filter(item => item.properties.alert === "-bass"), {
            pointToLayer : IconParseFEWS,
        });
        fewsStaBass.addTo(map);
        fewsStaBass.on('click', showPanelFews);

        fewsStaYl = L.geoJSON(layer.features.filter(item => item.properties.alert === "-Y"), {
            pointToLayer : IconParseFEWS,
        });
        fewsStaYl.addTo(map);
        fewsStaYl.on('click', showPanelFews);

        fewsStaOr = L.geoJSON(layer.features.filter(item => item.properties.alert === "-O"), {
            pointToLayer : IconParseFEWS,
        });
        fewsStaOr.addTo(map);
        fewsStaOr.on('click', showPanelFews);

        fewsStaRd = L.geoJSON(layer.features.filter(item => item.properties.alert === "-R"), {
            pointToLayer : IconParseFEWS,
        });
        fewsStaRd.addTo(map);
        fewsStaRd.on('click', showPanelFews);

    });


    // Load stations
    fetch("get-alerts")
    .then((response) => (layer = response.json()))
    .then((layer) => {
        
        est_R002 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R2"), {
            pointToLayer: IconParse,
        });
        est_R002.addTo(map);
        est_R002.on('click', showPanelAlert)
        
        est_R005 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R5"), {
            pointToLayer: IconParse,
        });
        est_R005.addTo(map);
        est_R005.on('click', showPanelAlert)

        est_R010 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R10"), {
            pointToLayer: IconParse,
        });
        est_R010.addTo(map);
        est_R010.on('click', showPanelAlert)

        est_R025 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R25"), {
            pointToLayer: IconParse,
        });
        est_R025.addTo(map);
        est_R025.on('click', showPanelAlert)

        est_R050 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R50"), {
            pointToLayer: IconParse,
        });
        est_R050.addTo(map);
        est_R050.on('click', showPanelAlert)

        est_R100 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R100"), {
            pointToLayer: IconParse,
        });
        est_R100.addTo(map);
        est_R100.on('click', showPanelAlert)


        // Filter by low alert
        est_L_1 = L.geoJSON(layer.features.filter(item => item.properties.alert === "lower_1"), {
            pointToLayer : IconParse,
        });
        est_L_1.addTo(map);
        est_L_1.on('click', showPanelAlert);

        est_L_3 = L.geoJSON(layer.features.filter(item => item.properties.alert === "lower_3"), {
            pointToLayer : IconParse,
        });
        est_L_3.addTo(map);
        est_L_3.on('click', showPanelAlert);

        est_L_7 = L.geoJSON(layer.features.filter(item => item.properties.alert === "lower_7"), {
            pointToLayer : IconParse,
        });
        est_L_7.addTo(map);
        est_L_7.on('click', showPanelAlert);

    });

}; 
