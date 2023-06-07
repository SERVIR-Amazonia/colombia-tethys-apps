from tethys_sdk.base import TethysAppBase


class NationalWaterLevelForecastColombia(TethysAppBase):
    """
    Tethys app class for National Water Level Forecast Colombia.
    """

    name = 'National Water Level Forecast Colombia'
    description = ''
    package = 'national_water_level_forecast_colombia'  # WARNING: Do not change this value
    index = 'home'
    icon = f'{package}/images/icon.gif'
    root_url = 'national-water-level-forecast-colombia'
    color = '#20295c'
    tags = ''
    enable_feedback = False
    feedback_emails = []