from src.normalize import name_core, normalize_address, normalize_name, postcodes


def test_legal_forms_are_canonicalised():
    assert normalize_name("Sharma Traders Private Limited") == normalize_name("Sharma Traders Pvt. Ltd.")
    assert name_core("Summit Hardware Corporation") == "summit hardware"


def test_ampersand_and_accents():
    assert normalize_name("Lumière & Fils SARL") == "lumiere and fils sarl"


def test_address_abbreviations_and_postcodes():
    assert normalize_address("12 MG Road") == normalize_address("12 MG Rd.")
    assert postcodes("5 Rue Rivoli, Paris 75001") == {"75001"}
    assert postcodes("Near SBI ATM, Pune 411001") == {"411001"}
