"""
Solar Image Source Configuration

Defines all available solar imaging sources from NOAA SWPC and Helioviewer API.
"""

# Helioviewer API base URL
HELIOVIEWER_API_URL = "https://api.helioviewer.org/v2/takeScreenshot/"

# All solar image sources organized by category
SOLAR_SOURCES = {
    "suvi": {
        "name": "GOES SUVI",
        "full_name": "GOES Solar Ultraviolet Imager",
        "description": "Real-time EUV images from NOAA's GOES-16/18 satellites. SUVI observes the Sun in six EUV wavelengths to monitor solar flares, coronal holes, and active regions.",
        "type": "noaa",
        "cadence": 60,  # seconds
        "images": [
            {
                "id": "suvi_094",
                "wavelength": "94A",
                "display_name": "94 Angstrom",
                "url": "https://services.swpc.noaa.gov/images/animations/suvi/primary/094/latest.png",
                "description": "Shows hot flare plasma (~6.3 million K). Best for detecting solar flares and very hot coronal loops."
            },
            {
                "id": "suvi_131",
                "wavelength": "131A",
                "display_name": "131 Angstrom",
                "url": "https://services.swpc.noaa.gov/images/animations/suvi/primary/131/latest.png",
                "description": "Shows flare plasma and active region cores (~10 million K during flares, ~400,000 K quiet)."
            },
            {
                "id": "suvi_171",
                "wavelength": "171A",
                "display_name": "171 Angstrom",
                "url": "https://services.swpc.noaa.gov/images/animations/suvi/primary/171/latest.png",
                "description": "Shows quiet corona and coronal loops (~600,000 K). Best for viewing coronal structure."
            },
            {
                "id": "suvi_195",
                "wavelength": "195A",
                "display_name": "195 Angstrom",
                "url": "https://services.swpc.noaa.gov/images/animations/suvi/primary/195/latest.png",
                "description": "Shows corona and hot active regions (~1.5 million K). Standard coronal imaging wavelength."
            },
            {
                "id": "suvi_284",
                "wavelength": "284A",
                "display_name": "284 Angstrom",
                "url": "https://services.swpc.noaa.gov/images/animations/suvi/primary/284/latest.png",
                "description": "Shows active regions and coronal holes (~2 million K). Good for identifying coronal holes."
            },
            {
                "id": "suvi_304",
                "wavelength": "304A",
                "display_name": "304 Angstrom",
                "url": "https://services.swpc.noaa.gov/images/animations/suvi/primary/304/latest.png",
                "description": "Shows chromosphere and transition region (~50,000 K). Best for prominences and filaments."
            },
        ]
    },
    "aia": {
        "name": "SDO AIA",
        "full_name": "Solar Dynamics Observatory - Atmospheric Imaging Assembly",
        "description": "High-resolution EUV and UV images from NASA's SDO spacecraft. AIA provides the highest resolution full-disk solar images available, updated every 12 seconds.",
        "type": "helioviewer",
        "cadence": 60,
        "images": [
            {
                "id": "aia_094",
                "wavelength": "94A",
                "display_name": "94 Angstrom",
                "sourceId": 8,
                "description": "Flare plasma at ~6.3 million K. Iron XVIII emission."
            },
            {
                "id": "aia_131",
                "wavelength": "131A",
                "display_name": "131 Angstrom",
                "sourceId": 9,
                "description": "Flare plasma (~10 MK) and transition region (~0.4 MK). Iron VIII, XX, XXIII."
            },
            {
                "id": "aia_171",
                "wavelength": "171A",
                "display_name": "171 Angstrom",
                "sourceId": 10,
                "description": "Quiet corona, coronal loops at ~600,000 K. Iron IX emission."
            },
            {
                "id": "aia_193",
                "wavelength": "193A",
                "display_name": "193 Angstrom",
                "sourceId": 11,
                "description": "Corona and hot flare plasma at ~1.2 MK and ~20 MK. Iron XII, XXIV."
            },
            {
                "id": "aia_211",
                "wavelength": "211A",
                "display_name": "211 Angstrom",
                "sourceId": 12,
                "description": "Active region corona at ~2 million K. Iron XIV emission."
            },
            {
                "id": "aia_304",
                "wavelength": "304A",
                "display_name": "304 Angstrom",
                "sourceId": 13,
                "description": "Chromosphere and transition region at ~50,000 K. Helium II emission."
            },
            {
                "id": "aia_335",
                "wavelength": "335A",
                "display_name": "335 Angstrom",
                "sourceId": 14,
                "description": "Active region corona at ~2.5 million K. Iron XVI emission."
            },
            {
                "id": "aia_1600",
                "wavelength": "1600A",
                "display_name": "1600 Angstrom",
                "sourceId": 15,
                "description": "Upper photosphere and transition region (~5,000 K and ~100,000 K). C IV + continuum."
            },
            {
                "id": "aia_1700",
                "wavelength": "1700A",
                "display_name": "1700 Angstrom",
                "sourceId": 16,
                "description": "Photosphere temperature minimum (~5,000 K). Continuum emission."
            },
            {
                "id": "aia_4500",
                "wavelength": "4500A",
                "display_name": "4500 Angstrom",
                "sourceId": 17,
                "description": "Photosphere (~6,000 K). White-light continuum, shows sunspots."
            },
        ]
    },
    "hmi": {
        "name": "SDO HMI",
        "full_name": "Solar Dynamics Observatory - Helioseismic and Magnetic Imager",
        "description": "Measures photospheric magnetic fields and visible-light intensity from NASA's SDO. Essential for tracking sunspots and active region magnetic structure.",
        "type": "helioviewer",
        "cadence": 60,
        "images": [
            {
                "id": "hmi_continuum",
                "wavelength": "Continuum",
                "display_name": "Continuum Intensity",
                "sourceId": 18,
                "description": "White-light image showing the photosphere. Best for viewing sunspots and solar granulation."
            },
            {
                "id": "hmi_magnetogram",
                "wavelength": "Magnetogram",
                "display_name": "Magnetogram",
                "sourceId": 19,
                "description": "Line-of-sight magnetic field. White = positive (north) polarity, Black = negative (south) polarity."
            },
        ]
    },
    "lasco": {
        "name": "SOHO LASCO",
        "full_name": "Solar and Heliospheric Observatory - Large Angle Spectrometric Coronagraph",
        "description": "White-light coronagraph images from ESA/NASA's SOHO spacecraft at L1. Essential for detecting coronal mass ejections (CMEs) propagating into the heliosphere.",
        "type": "helioviewer",
        "cadence": 900,  # 15 minutes - LASCO updates less frequently
        "images": [
            {
                "id": "lasco_c2",
                "wavelength": "C2",
                "display_name": "C2 Coronagraph",
                "sourceId": 4,
                "imageScale": 11.5,  # C2: 2-6 solar radii, needs ~12 Rs total FOV
                "description": "Inner coronagraph (2-6 solar radii). Shows inner corona and CME launch. Orange occulting disk blocks the bright solar disk."
            },
            {
                "id": "lasco_c3",
                "wavelength": "C3",
                "display_name": "C3 Coronagraph",
                "sourceId": 5,
                "imageScale": 56.0,  # C3: 4-30 solar radii, needs ~60 Rs total FOV
                "description": "Outer coronagraph (4-30 solar radii). Shows CME propagation into heliosphere. Blue occulting disk, wider field of view."
            },
        ]
    },
    "eit": {
        "name": "SOHO EIT",
        "full_name": "Solar and Heliospheric Observatory - Extreme ultraviolet Imaging Telescope",
        "description": "EUV images from SOHO at L1. Lower resolution than SDO/AIA but provides a complementary view from L1 (1.5 million km from Earth).",
        "type": "helioviewer",
        "cadence": 900,  # EIT updates less frequently
        "images": [
            {
                "id": "eit_171",
                "wavelength": "171A",
                "display_name": "171 Angstrom",
                "sourceId": 0,
                "description": "Quiet corona at ~1 million K. Iron IX/X emission. Green colormap."
            },
            {
                "id": "eit_195",
                "wavelength": "195A",
                "display_name": "195 Angstrom",
                "sourceId": 1,
                "description": "Corona at ~1.5 million K. Iron XII emission. Standard coronal imaging."
            },
            {
                "id": "eit_284",
                "wavelength": "284A",
                "display_name": "284 Angstrom",
                "sourceId": 2,
                "description": "Active regions at ~2 million K. Iron XV emission. Yellow colormap."
            },
            {
                "id": "eit_304",
                "wavelength": "304A",
                "display_name": "304 Angstrom",
                "sourceId": 3,
                "description": "Chromosphere/transition region at ~80,000 K. Helium II emission. Orange/red colormap."
            },
        ]
    },
}


def get_all_sources():
    """Return a flat list of all image sources with category info."""
    all_sources = []
    for category_key, category in SOLAR_SOURCES.items():
        for img in category['images']:
            source = img.copy()
            source['category'] = category_key
            source['category_name'] = category['name']
            source['category_full_name'] = category['full_name']
            source['type'] = category['type']
            source['cadence'] = category['cadence']
            if 'url' not in source:
                source['url'] = None  # Helioviewer sources don't have direct URLs
            all_sources.append(source)
    return all_sources


def get_source_by_id(source_id: str):
    """Get a specific source by its ID."""
    for source in get_all_sources():
        if source['id'] == source_id:
            return source
    return None
