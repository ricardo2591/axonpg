"""Tests for multilingual Unicode normalization, tokenization, and CLI localization."""

from __future__ import annotations

from axonpg.i18n import (
    _MESSAGES,
    SUPPORTED_LANGUAGES,
    extract_search_tokens,
    fold_accents_preserving_scripts,
    normalize_lookup_key,
    normalize_unicode,
    resolve_language,
    t,
)


def test_translations_cover_all_supported_languages():
    expected_keys = list(_MESSAGES["en"].keys())
    assert expected_keys
    for lang in SUPPORTED_LANGUAGES:
        for key in expected_keys:
            val = _MESSAGES[lang].get(key)
            assert val and val != key, f"Missing translation for '{key}' in '{lang}'"
    assert t("step_of", "es", step=2, total=6) == "Paso 2/6"
    assert t("preflight_port_free", "fr", port=5435) == "port 5435 libre"


def test_resolve_language_handles_bcp47_and_posix_tags():
    assert resolve_language("es-CO") == "es"
    assert resolve_language("pt_BR") == "pt"
    assert resolve_language("fr-FR") == "fr"
    assert resolve_language("ja-JP") == "en"  # UI falls back cleanly to English while data stays all-script


def test_unicode_nfc_and_accent_folding_preserve_non_latin_scripts():
    decomposed = "Cafe\u0301"
    assert normalize_unicode(decomposed) == "Café"

    assert fold_accents_preserving_scripts("Configuración de Camión") == "Configuracion de Camion"
    assert fold_accents_preserving_scripts("Ação em São Paulo") == "Acao em Sao Paulo"
    assert fold_accents_preserving_scripts("Réseau résumé") == "Reseau resume"

    # Non-Latin scripts stay intact (not corrupted or erased like ASCII transliteration does)
    cjk = fold_accents_preserving_scripts("東京サーバー 設定")
    assert "東京サーバー" in cjk and "設定" in cjk

    arabic = fold_accents_preserving_scripts("مراجعة الخادم")
    assert "مراجعة" in arabic and "الخادم" in arabic

    cyrillic = fold_accents_preserving_scripts("Сервер Базы Данных")
    assert cyrillic == "Сервер Базы Данных"


def test_lookup_key_and_multilingual_tokens():
    assert normalize_lookup_key("  Deploy::PostgreSQL\tCamión  ") == "deploy_postgresql_camion"
    tokens = extract_search_tokens("¡Error en São Paulo! 東京123")
    assert "error" in tokens
    assert "sao" in tokens
    assert "paulo" in tokens
    assert any("東京" in tok for tok in tokens)
