from dataclasses import dataclass

@dataclass
class PageQualityAssessment:
    page_number: int

    quality_class: str

    native_text_usable: bool
    ocr_required: bool
    suspicious_text_layer: bool

    appears_scanned: bool
    has_large_page_image: bool

    native_character_count: int
    native_word_count: int
    native_block_count: int

    image_count: int
    maximum_image_coverage: float

    visual_contrast: float
    visual_sharpness: float

    low_contrast: bool
    low_sharpness: bool

    reasons: list[str]
