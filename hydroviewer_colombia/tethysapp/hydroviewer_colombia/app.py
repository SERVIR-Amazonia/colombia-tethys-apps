from tethys_sdk.base import TethysAppBase


class HydroviewerColombia(TethysAppBase):
    """
    Tethys app class for Hydroviewer Colombia.
    """

    name = 'Hydroviewer Colombia'
    description = ''
    package = 'hydroviewer_colombia'  # WARNING: Do not change this value
    index = 'home'
    icon = f'{package}/images/icon.gif'
    root_url = 'hydroviewer-colombia'
    color = '#20295c'
    tags = ''
    enable_feedback = False
    feedback_emails = []