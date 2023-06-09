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

// Icon markers for each return period
const IconR000 = IconMarker("0");       // RP: 0 years
const IconR002 = IconMarker("2");      // RP: 2 years
const IconR005 = IconMarker("5");        // RP: 5 years
const IconR010 = IconMarker("10");      // RP: 10 years
const IconR025 = IconMarker("25");         // RP: 25 years
const IconR050 = IconMarker("50");      // RP: 50 years
const IconR100 = IconMarker("100");       // RP: 100 years

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
    }
    return L.marker(latlng, { icon: StationIcon });
}




// ------------------------------------------------------------------------------------------------------------ //
//                                            PANEL DATA INFORMATION                                            //
// ------------------------------------------------------------------------------------------------------------ //
const sleep = ms => new Promise(r => setTimeout(r, ms));

var global_comid;

async function get_data_station(comid, name, lat, lon, loc1, loc2){
    // Updating the comid
    global_comid = comid
    
    // Add data to the panel
    $("#station-comid-custom").html(`<b>COMID:</b> &nbsp ${comid}`)
    $("#station-river-custom").html(`<b>RIO:</b> &nbsp ${name}`)
    $("#station-latitude-custom").html(`<b>LATITUD:</b> &nbsp ${lat}`)
    $("#station-longitude-custom").html(`<b>LONGITUD:</b> &nbsp ${lon}`)
    $("#station-locality1-custom").html(`<b>ZONA HIDROGRÁFICA:</b> &nbsp ${loc1}`)
    $("#station-locality2-custom").html(`<b>SUBZONA HIDROGRÁFICA:</b> &nbsp ${loc2}`)

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
  function showPanel(e) {
    var comid = e.layer.feature.properties.comid;
    var name = e.layer.feature.properties.river;
    var lat = e.layer.feature.properties.latitude;
    var lon = e.layer.feature.properties.longitude;
    var loc1 = e.layer.feature.properties.loc1;
    var loc2 = e.layer.feature.properties.loc2;
    $("#panel-modal").modal("show")
    get_data_station(comid, name, lat, lon, loc1, loc2)
  }


  // Load drainage network
  fetch(
    `${server}/static/hydroviewer_colombia/geojson/colombia_geoglows_drainage.geojson`
  )
    .then((response) => (layer = response.json()))
    .then((layer) => {
      // Adding the drainage network to the map
      riv = L.geoJSON(layer, {
        style: {
          weight: 1, 
          color: "#4747C9" 
        }
      }).addTo(map);

      // Fit the map to the river bounds
      map.fitBounds(riv.getBounds());

      // Buffer to select rivers
      map.almostOver.addLayer(riv);

      // On click function
      map.on('almost:click', showPanel);
    });

  // Retrieve the alets

  
// Load stations
fetch("get-alerts")
  .then((response) => (layer = response.json()))
  .then((layer) => {
    
      est_R002 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R2"), {
          pointToLayer: IconParse,
      });
      est_R002.addTo(map);
      est_R002.on('click', showPanel)
      
      est_R005 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R5"), {
          pointToLayer: IconParse,
      });
      est_R005.addTo(map);
      est_R005.on('click', showPanel)

      est_R010 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R10"), {
          pointToLayer: IconParse,
      });
      est_R010.addTo(map);
      est_R010.on('click', showPanel)

      est_R025 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R25"), {
          pointToLayer: IconParse,
      });
      est_R025.addTo(map);
      est_R025.on('click', showPanel)

      est_R050 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R50"), {
          pointToLayer: IconParse,
      });
      est_R050.addTo(map);
      est_R050.on('click', showPanel)

      est_R100 = L.geoJSON(layer.features.filter(item => item.properties.alert === "R100"), {
          pointToLayer: IconParse,
      });
      est_R100.addTo(map);
      est_R100.on('click', showPanel)

  });


}; 
 
 