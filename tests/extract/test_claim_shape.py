from infovore.extract.claim_shape import (
    attributes_to_a_speaker,
    has_person_subject,
    shape_counts,
)

# Real statements from the live claims table, issue #165.
PERSON_SUBJECT = (
    "A community member reported that their SGI O2 is noticeably quieter than their Octane.",
    "A community member paid $250 for an SGI Octane2 with dual 600MHz CPUs and V12 graphics.",
    "A community member speculated it is likely possible to configure a Tezro power button.",
    "Someone found that the DMediaPro card weighs over 100 lbs when boxed.",
    "The author stated that 68k IRIS source code has no practical use without the hardware.",
)

THING_SUBJECT = (
    "The SGI O2 power supply can be substituted with a Meanwell modular switching unit.",
    "tartube's GTK3 interface runs sluggishly on IRIX, observed on an SGI Octane.",
    "IRIX 6.5.22m requires patchSG0007188 before the MIPSpro 7.4.4m front-end will install.",
    "An SGI Tezro with 4x 1.0 GHz R16000 processors reports IP35 in hinv.",
    "The VBOB video breakout box requires either a DM2 or DM3 card to function.",
)


def test_person_subject_fires_on_real_attributed_claims() -> None:
    assert all(has_person_subject(text) for text in PERSON_SUBJECT)


def test_person_subject_does_not_fire_on_claims_about_things() -> None:
    assert not any(has_person_subject(text) for text in THING_SUBJECT)


def test_a_person_later_in_the_sentence_is_not_the_subject() -> None:
    """The subject is what the claim is ABOUT. A trailing attribution is
    a provenance nit, not the defect #165 is measuring."""
    text = "The O2 PSU can be replaced with a Meanwell unit, which a community member confirmed."

    assert not has_person_subject(text)
    assert attributes_to_a_speaker(text)


def test_attribution_catches_according_to() -> None:
    assert attributes_to_a_speaker("According to a member, the Indy maxes at 256MB.")


def test_attribution_does_not_fire_on_a_bare_thing_claim() -> None:
    assert not any(attributes_to_a_speaker(text) for text in THING_SUBJECT)


def test_a_person_noun_with_no_attribution_verb_is_not_attribution() -> None:
    """'Members of the community' as part of a fact is not the pipeline
    transcribing an utterance."""
    text = "The sgug-rse repository is maintained by a member of the community."

    assert not attributes_to_a_speaker(text)


def test_shape_counts_reports_both_shares() -> None:
    counts = shape_counts([*PERSON_SUBJECT, *THING_SUBJECT])

    assert counts.total == 10
    assert counts.person_subject == 5
    assert counts.person_subject_share == 0.5


def test_shape_counts_of_nothing_is_zero_not_a_crash() -> None:
    counts = shape_counts([])

    assert counts.total == 0
    assert counts.person_subject_share == 0.0
    assert counts.attributed_share == 0.0
