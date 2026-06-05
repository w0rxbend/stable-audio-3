#!/usr/bin/env python3
"""
generate_from_lofi_presets.py — generate audio via stable-audio CLI from lofi-maker presets.

Reads YAML preset files from ../twitch-musicplayer/presets/lofi/ (or --presets-dir),
translates each preset's mood, instruments, BPM, scale, and effects into a rich text
prompt, then invokes the stable-audio CLI for each track.

Prerequisites:
  - HuggingFace account with access to the target model (see --model).
    Request access at:
      https://huggingface.co/stabilityai/stable-audio-3-medium         (for medium)
      https://huggingface.co/stabilityai/stable-audio-3-small-music   (for small-music)
  - Logged in:  huggingface-cli login   (or HF_TOKEN env var)
  - stable-audio-3 installed:  uv sync  (from within ../stable-audio-3/)
  - pyyaml available:  uv pip install pyyaml

Basic usage:
  # single preset
  python generate_from_lofi_presets.py --preset ambient_lofi

  # all presets, one track each
  python generate_from_lofi_presets.py --all --duration 240

  # random preset, 3 variations with prompt randomization
  python generate_from_lofi_presets.py --random --count 3 --seed 42 --randomize-prompts

  # generate directly from the 200-prompt bank (ignores preset parameters)
  python generate_from_lofi_presets.py --random-from-bank --count 5

  # dry-run: see prompts and commands without generating
  python generate_from_lofi_presets.py --all --dry-run

  # list available presets
  python generate_from_lofi_presets.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "Error: pyyaml is required.\n"
        "  Install with:  uv pip install pyyaml\n"
        "  Or activate the venv first:  source .venv/bin/activate",
        file=sys.stderr,
    )
    sys.exit(1)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESETS_DIR = _SCRIPT_DIR.parent / "twitch-musicplayer" / "presets" / "lofi"

# Human-readable labels for instrument names used in preset YAML files.
_INSTRUMENT_LABELS: dict[str, str] = {
    "felt_piano": "felt piano",
    "rhodes": "Rhodes electric piano",
    "pad": "lush synth pad",
    "muted_trumpet": "muted trumpet",
    "upright_bass": "upright bass",
    "muted_bass": "muted bass",
    "soft_kit": "soft drum kit",
    "brush_kit": "jazz brush drums",
    "dusty_808": "808 drum machine",
    "piano": "piano",
    "electric_piano": "electric piano",
    "marimba": "marimba",
    "vibraphone": "vibraphone",
    "acoustic_guitar": "acoustic guitar",
    "guitar": "guitar",
}

# Negative prompt applied to every generation — suppresses fast/harsh/vocal elements.
DEFAULT_NEGATIVE_PROMPT = (
    "fast tempo, harsh distortion, heavy metal, EDM, house music, "
    "techno, screaming vocals, lead singing, aggressive percussion, noise, glitch"
)

# ---------------------------------------------------------------------------
# 200-prompt bank  (used by --random-from-bank and optionally as atmosphere
# injections via --randomize-prompts)
# ---------------------------------------------------------------------------

LOFI_PROMPT_BANK: tuple[str, ...] = (
    # Rain & weather (1–15)
    "lofi hip hop instrumental, soft felt piano chords, gentle rain on glass, muted bass, brushed snare, vinyl warmth, 70 BPM, minor key, wide reverb, tape hiss, no vocals, loopable",
    "lofi chill, thunderstorm ambience, Rhodes electric piano, upright bass, 68 BPM, A minor, heavy reverb, tape saturation, melancholy mood, no vocals",
    "cozy rainy-day lofi, mellow felt piano, soft drum kit, warm sub-bass, vinyl crackle, warm low-pass filter, 75 BPM, minor pentatonic, nostalgic, no vocals",
    "lofi hip hop, monsoon season, felt piano with sustain pedal, jazz brushes, upright bass pluck, 66 BPM, F minor, room reverb, tape warmth, introspective, no vocals",
    "rainy window lofi, Rhodes piano chords, sparse piano accents, muted bass, 72 BPM, minor key, moderate reverb, no drums, no vocals",
    "midnight rain lofi, soft classical piano, falling droplets, sparse bass notes, 60 BPM, D minor, cathedral reverb, very slow, meditative, no vocals",
    "lofi hip hop rainy night, muted trumpet melody, Rhodes chords, upright bass, jazz brushes, 78 BPM, minor key, vinyl crackle, warm saturation, no vocals",
    "grey morning lofi, felt piano, steady rain ambience, soft hi-hat, muted bass, 73 BPM, E minor, warm low-pass filter, tape wow, cozy indoors, no vocals",
    "lofi rainy season, acoustic guitar fingerpicking, soft kit, muted bass, 70 BPM, C minor, wide reverb, gentle tape hiss, melancholy but peaceful, no vocals",
    "lofi ambient rain, sparse vibraphone notes, bass pad, gentle rain texture, 58 BPM, no percussion, G minor, huge reverb, floating atmosphere, no vocals",
    "lofi hip hop in the rain, piano seventh chords, upright bass, brush drums, 76 BPM, A flat minor, vinyl warmth, warm low-pass filter, no vocals",
    "stormy lofi, heavy reverb piano, deep bass drones, distant thunder texture, 65 BPM, minor key, atmospheric, no percussion, no vocals",
    "lofi drizzle session, felt piano melody, soft jazz drums, walking bass, 80 BPM, B flat minor, cassette warmth, moderate reverb, no vocals",
    "rainy afternoon lofi, electric piano chords, sparse piano melody, muted bass, 69 BPM, minor key, warm room reverb, vinyl texture, no vocals",
    "lofi autumn rain, piano with reverb bloom, soft shuffle kit, bass, 74 BPM, G minor, late-night mood, tape warmth, no vocals",
    # Late night city (16–30)
    "lofi hip hop late night, city traffic ambience, Rhodes piano, walking bass, brush kit, 82 BPM, minor key, vinyl crackle, warm saturation, no vocals",
    "3 AM lofi, felt piano, distant sirens, muted bass, sparse hi-hat, 70 BPM, D minor, heavy reverb, introspective, no vocals",
    "late-night study lofi, Rhodes chords, soft piano melody, muted bass, gentle drums, 76 BPM, A minor, warm low-pass filter, tape hiss, cozy, no vocals",
    "lofi city night, jazz-inflected piano, upright bass, brush snare, 84 BPM, minor key, vinyl warmth, moderate reverb, nostalgic, no vocals",
    "midnight city lofi, muted piano, traffic hum, soft kit, bass groove, 78 BPM, E minor, saturation, warm low-pass, no vocals",
    "lofi 2 AM session, electric piano, sparse chord stabs, bass, shuffling drums, 80 BPM, minor key, tape warmth, vinyl crackle, no vocals",
    "neon city lofi, Rhodes piano, glowing bass, soft hi-hat groove, 85 BPM, F minor, warm saturation, light reverb, urban, no vocals",
    "lofi hip hop night drive, piano chords, deep bass, jazz drums, 88 BPM, C minor, cassette texture, moderate reverb, no vocals",
    "quiet city lofi, felt piano solo lines, muted bass, sparse brushes, 66 BPM, minor key, wide room reverb, melancholy, no vocals",
    "late night jazz lofi, piano comp, upright bass, brushed snare, 75 BPM, B flat minor, vinyl warmth, intimate room reverb, no vocals",
    "city lights lofi, Rhodes with tremolo, bass, soft kit, 82 BPM, A minor, warm low-pass filter, tape wow, nostalgic, no vocals",
    "lofi streetlamp, piano, bass, warm drums, 77 BPM, minor key, vinyl crackle, moderate reverb, urban melancholy, no vocals",
    "sleepless night lofi, slow piano, drone bass, no drums, 58 BPM, D minor, heavy reverb, floating texture, no vocals",
    "lofi midnight cafe, piano trio, walking bass, gentle brushes, 80 BPM, G minor, warm reverb, analog tape texture, no vocals",
    "lofi night shift, muted electric piano, bass, sparse percussion, 72 BPM, minor key, cassette warmth, warm low-pass filter, no vocals",
    # Piano-led (31–50)
    "lofi solo piano, gentle felt piano melody, soft bass notes, no drums, 65 BPM, C major, wide reverb, tape hiss, introspective, no vocals",
    "lofi piano ballad, felt piano chords and melody, muted bass, very sparse brushes, 68 BPM, D minor, heavy reverb, gentle dynamics, no vocals",
    "lofi hip hop piano, bouncy felt piano melody, bass groove, soft kit, 80 BPM, minor key, vinyl warmth, warm low-pass filter, no vocals",
    "morning lofi piano, bright felt piano, warm bass, gentle shuffle drums, 78 BPM, G major, moderate reverb, cassette texture, hopeful, no vocals",
    "lofi piano meditation, slow chord progressions, sparse bass, no drums, 60 BPM, E minor, cathedral reverb, floating, no vocals",
    "piano cafe lofi, felt piano with light touch, upright bass, brushes, 75 BPM, A major, warm room reverb, analog warmth, no vocals",
    "lofi bedroom piano, simple piano melody, muted bass, soft kit, 72 BPM, F major, moderate reverb, tape hiss, cozy, no vocals",
    "lofi piano at night, slow minor piano, deep bass drones, no percussion, 62 BPM, C minor, enormous reverb, ethereal, no vocals",
    "upright piano lofi, raw acoustic piano tone, brushed snare, bass, 82 BPM, D minor, room mic reverb, slight distortion, no vocals",
    "lofi neo-classical piano, intricate piano arpeggios, cello-like bass, no drums, 70 BPM, F minor, reverb hall, delicate, no vocals",
    "piano and rain lofi, felt piano with reverb, rain texture, muted bass, no drums, 64 BPM, B minor, lush reverb, melancholy mood, no vocals",
    "lofi piano sketch, improvised piano lines, bass, gentle rim shot, 76 BPM, A minor, tape warmth, moderate reverb, spontaneous, no vocals",
    "jazz-touched lofi piano, piano comping chords, muted trumpet, upright bass, brushes, 85 BPM, minor key, vinyl crackle, warm, no vocals",
    "sleepy piano lofi, hushed felt piano, minimal bass, very sparse drums, 67 BPM, G minor, heavy reverb, intimate, no vocals",
    "lofi piano dawn, slowly building piano chords, bass, soft kit, 73 BPM, E major, warm reverb, hopeful, morning mood, no vocals",
    "lofi piano nostalgia, old-style piano tone, shuffling drums, bass, 78 BPM, C major, vinyl warmth, cassette hiss, nostalgic, no vocals",
    "piano pad lofi, piano chords held with lush pads, deep bass, no drums, 60 BPM, D major, cavernous reverb, ambient blend, no vocals",
    "lofi hip hop keys, punchy piano stabs, bass groove, soft kit, 90 BPM, minor key, vinyl warmth, warm low-pass, no vocals",
    "lofi piano encore, delicate piano figurations, bass, brushes, 74 BPM, F minor, room reverb, intimate, no vocals",
    "meditative lofi piano, long sustained piano notes, bass drone, 55 BPM, C minor, infinite reverb, no percussion, ambient, no vocals",
    # Jazz-inflected (51–70)
    "lofi jazz, upright bass walking, piano comp, brush snare, muted trumpet melody, 88 BPM, G minor, vinyl crackle, warm saturation, no vocals",
    "lofi bossa nova, acoustic guitar chords, acoustic bass, shaker, piano accents, 90 BPM, D major, light reverb, warm, no vocals",
    "lofi cool jazz, piano trio, walking bass, brushed cymbals, 82 BPM, A minor, vintage vinyl warmth, no vocals",
    "lofi hip hop jazz, Rhodes piano, upright bass, jazz brushes, 80 BPM, F minor, tape warmth, moderate reverb, no vocals",
    "lofi swing, piano chords with swing feel, bass, brushed snare, 85 BPM, B flat minor, vinyl crackle, nostalgic jazz club, no vocals",
    "late-night jazz lofi, muted piano, bass, soft cymbals, 76 BPM, E flat minor, warm reverb, intimate, no vocals",
    "lofi jazz trio, piano, bass, drums in conversation, 84 BPM, C minor, vintage tape texture, moderate reverb, no vocals",
    "lofi bebop piano, chord changes, walking bass, brushes, 92 BPM, minor key, vinyl warmth, cassette hiss, no vocals",
    "lofi jazz ballade, slow piano, bowed bass, no drums, 60 BPM, D flat minor, cathedral reverb, melancholy, no vocals",
    "chillhop jazz, Rhodes piano comp, upright bass pluck, brushed snare, 80 BPM, minor key, analog warmth, light reverb, no vocals",
    "lofi trumpet and piano, muted trumpet lines, Rhodes comp, bass, brushes, 78 BPM, A minor, vinyl warmth, warm reverb, no vocals",
    "lofi jazz nocturne, piano, bass drone, ambient brushes, 65 BPM, G minor, heavy reverb, atmospheric, no vocals",
    "sunny lofi jazz, piano trio, upbeat bass line, gentle drums, 88 BPM, D major, warm saturation, light reverb, hopeful, no vocals",
    "lofi post-bop piano, complex chord voicings, bass, drums, 84 BPM, minor key, vintage vinyl texture, no vocals",
    "lofi jazz waltz, piano in 3/4, bass, brushes, 72 BPM, A minor, warm room reverb, gentle swing, no vocals",
    "lofi impressionist jazz, Ravel-like piano, bowed bass, sparse percussion, 68 BPM, F major, lush reverb, ethereal, no vocals",
    "lofi vibes and piano, vibraphone melody, piano comp, bass, brushes, 80 BPM, minor key, vinyl warmth, moderate reverb, no vocals",
    "lofi jazz lullaby, gentle piano, bass, very soft brushes, 66 BPM, C minor, warm reverb, intimate, soothing, no vocals",
    "chillhop jazz fusion, electric piano, bass, light drums, 85 BPM, D minor, cassette texture, moderate reverb, no vocals",
    "lofi jazz standard, piano reharmonized, upright bass, brushed snare, 80 BPM, G minor, vinyl crackle, nostalgic, no vocals",
    # Dreamy / ambient pads (71–90)
    "ambient lofi, lush synth pads, felt piano accents, no drums, 60 BPM, C minor, enormous reverb, floating, dreamy, no vocals",
    "lofi dreamscape, slow pad swells, gentle piano, bass drone, 55 BPM, G minor, cathedral reverb, no percussion, immersive, no vocals",
    "lofi ambient hip hop, pads and piano, soft sub-bass, very sparse drums, 68 BPM, minor key, wide reverb, tape warmth, no vocals",
    "lofi cloud hop, soft pads, distant piano notes, bass, gentle hi-hat, 70 BPM, D minor, lush reverb, dreamy, no vocals",
    "ambient lofi pads, evolving pad textures, piano chord stabs, bass, 58 BPM, F minor, reverb-heavy, hypnotic, no vocals",
    "lofi space ambient, slow pads, Rhodes notes, deep bass, no drums, 52 BPM, A minor, cavernous reverb, cosmic, no vocals",
    "dream lofi, soft lush pads, gentle piano melody, muted bass, 64 BPM, E minor, wide reverb, slow attack, no vocals",
    "lofi ambient meditation, long pad notes, occasional piano, bass drone, 50 BPM, C minor, infinite reverb, healing, no vocals",
    "lofi lo-brow ambient, pads with vinyl texture, piano touches, bass, 60 BPM, G major, moderate reverb, warm, no vocals",
    "ambient lofi hip hop, suspended pads, piano arpeggios, bass groove, soft kit, 72 BPM, minor key, reverb blend, tape warmth, no vocals",
    "lofi slow dance, pads, gentle Rhodes, bass, brushes, 65 BPM, D minor, warm reverb, intimate, no vocals",
    "lofi ambient deep, evolving textures, piano accents, sub-bass, 55 BPM, A minor, reverb hall, cinematic, no vocals",
    "lofi vapor ambient, slowed pads, piano texture, bass, 60 BPM, F minor, heavy reverb, surreal, no vocals",
    "soft lofi ambient, gentle pads, felt piano, upright bass, 68 BPM, C major, moderate reverb, hopeful, no vocals",
    "lofi ambient night sky, wide pad chords, slow piano melody, no drums, 58 BPM, B minor, vast reverb, no vocals",
    "lofi soundscape, textured pads, piano chord stab, bass, 62 BPM, E minor, enormous reverb, immersive, no vocals",
    "lofi ambient glow, warm brass-like pads, piano, bass, 65 BPM, G minor, room reverb, golden hour, no vocals",
    "lofi ambient lullaby, soft string pads, piano, bass drone, 55 BPM, D minor, vast reverb, sleeping mood, no vocals",
    "lofi ambient waves, ocean-like pad swells, piano accents, bass, 58 BPM, A minor, reverb wash, meditative, no vocals",
    "lofi pad journey, evolving pad motion, piano melody, bass, sparse rim shots, 64 BPM, C minor, wide reverb, traveling, no vocals",
    # Study / focus (91–105)
    "lofi study session, focused piano melody, moderate bass, soft kit, 80 BPM, minor key, warm low-pass filter, vinyl warmth, productive, no vocals",
    "lofi hip hop study beats, Rhodes piano, bass, drum groove, 85 BPM, D minor, cassette texture, moderate reverb, no vocals",
    "exam time lofi, gentle piano, bass, light drums, 76 BPM, A minor, warm room reverb, focused, no vocals",
    "library lofi, hushed piano, bass, soft hi-hat, 74 BPM, G minor, moderate reverb, quiet atmosphere, no vocals",
    "lofi productivity, upbeat piano comp, bass groove, drum loop, 88 BPM, minor key, vinyl crackle, energetic but calm, no vocals",
    "midnight study lofi, piano and bass, light drums, 78 BPM, C minor, tape warmth, warm low-pass filter, no vocals",
    "early morning study lofi, bright piano, bass, gentle drum loop, 82 BPM, D major, warm reverb, hopeful, no vocals",
    "lofi focus flow, hypnotic piano pattern, bass, steady drum groove, 84 BPM, minor key, vinyl texture, moderate reverb, no vocals",
    "deep work lofi, minimal piano, bass, soft drums, 78 BPM, A minor, warm low-pass filter, cassette hiss, no vocals",
    "lofi coding beats, punchy piano chords, bass groove, drum loop, 90 BPM, minor key, warm saturation, vinyl crackle, no vocals",
    "lofi study cafe, piano trio, upright bass, brushes, 80 BPM, G minor, warm room reverb, cafe ambience, no vocals",
    "creative flow lofi, fluid piano melody, bass, light kit, 82 BPM, D minor, tape warmth, moderate reverb, no vocals",
    "focus lofi beats, electric piano, bass, shuffling drums, 86 BPM, minor key, analog warmth, no vocals",
    "lofi writing session, gentle piano, bass, very soft drums, 74 BPM, F minor, wide reverb, quiet, no vocals",
    "lofi homework beats, piano stabs, bass, drum loop, 85 BPM, A minor, vinyl warmth, light reverb, no vocals",
    # Nostalgic / tape (106–120)
    "vintage lofi, worn tape piano, crackly vinyl, bass, old-school drum loop, 80 BPM, D minor, saturated, heavy vinyl crackle, no vocals",
    "cassette lofi, muted piano through tape, bass, dusty drums, 76 BPM, G minor, tape saturation, warm low-pass filter, nostalgic, no vocals",
    "lofi throwback, lo-fi piano, bass groove, old-school drums, 82 BPM, minor key, heavy vinyl noise, warm saturation, no vocals",
    "retro lofi hip hop, sample-style piano chop, bass, drum break, 88 BPM, C minor, cassette warmth, filtered, no vocals",
    "lofi golden era, boom-bap piano, bass, classic drum loop, 86 BPM, A minor, vinyl crackle, moderate reverb, no vocals",
    "bedroom producer lofi, piano sample, bass, sampled drums, 80 BPM, minor key, tape hiss, warm, no vocals",
    "lofi 90s vibe, Rhodes piano, bass, head-nod drum groove, 84 BPM, D minor, vinyl warmth, warm low-pass filter, no vocals",
    "old cassette lofi, degraded piano texture, bass, fuzzy drums, 76 BPM, minor key, heavy tape noise, lo-fi treatment, no vocals",
    "lofi boom bap, chopped piano, bass, break-beat drums, 90 BPM, G minor, vinyl crackle, classic hip hop feel, no vocals",
    "lofi time machine, vintage piano sound, bass, swinging drums, 82 BPM, F minor, analog texture, nostalgic, no vocals",
    "lofi memory tape, warm piano through cassette, bass, soft kit, 75 BPM, minor key, tape hiss, warm saturation, no vocals",
    "lofi lost tapes, distressed piano, bass, lo-fi drum samples, 78 BPM, C minor, heavy vinyl noise, gritty, no vocals",
    "lofi record shop, piano over wax, bass, drum loop, 84 BPM, A minor, thick vinyl crackle, vintage, no vocals",
    "lofi hip hop roots, classic piano chop, bass groove, boom-bap kit, 88 BPM, minor key, cassette warmth, no vocals",
    "lofi archive, warm old piano, bass, sampled percussion, 78 BPM, G minor, vinyl saturation, nostalgic, no vocals",
    # Nature / outdoor (121–135)
    "lofi forest session, piano in nature, birdsong texture, bass, soft kit, 72 BPM, D major, wide reverb, fresh air, no vocals",
    "morning dew lofi, gentle piano, soft bass, no drums, birds in distance, 65 BPM, G major, room reverb, hopeful, no vocals",
    "lofi by the creek, piano, water sounds, bass, brushes, 70 BPM, C major, natural reverb, peaceful, no vocals",
    "lofi open air, outdoor piano texture, wide pads, bass, 60 BPM, E minor, vast reverb, expansive, no vocals",
    "autumn leaves lofi, felt piano, rustling texture, bass, soft kit, 74 BPM, A minor, warm reverb, melancholy, no vocals",
    "lofi ocean, piano, distant waves, bass drone, no drums, 58 BPM, C minor, ocean reverb, vast, no vocals",
    "lofi mountain air, crisp piano, wide pads, bass, no percussion, 60 BPM, E major, natural hall reverb, fresh, no vocals",
    "garden lofi, piano and acoustic guitar, bass, brush kit, 78 BPM, G major, warm room reverb, peaceful, no vocals",
    "lofi campfire, warm piano, crackling fire texture, bass, sparse brushes, 68 BPM, A minor, intimate reverb, cozy outdoor, no vocals",
    "lofi meadow, bright piano, soft string-like pads, bass, 72 BPM, D major, wide reverb, warm sunlight, no vocals",
    "lofi seaside, Rhodes piano, ocean ambience, upright bass, brushes, 76 BPM, minor key, reverb blend, relaxed, no vocals",
    "forest floor lofi, pad textures, slow piano, deep bass, 55 BPM, F minor, enormous reverb, mystical, no vocals",
    "lofi sunrise, ascending piano chords, soft pads, bass, very soft kit, 68 BPM, C major, warm reverb, hopeful, no vocals",
    "lofi countryside, acoustic guitar, piano, bass, brushes, 80 BPM, G major, natural reverb, bright, no vocals",
    "lofi cherry blossoms, light piano, spring texture, bass, gentle kit, 73 BPM, E major, delicate reverb, serene, no vocals",
    # Melancholy / emotional (136–155)
    "deeply melancholy lofi, slow piano in minor, bass drone, no drums, 58 BPM, D minor, cathedral reverb, heartbroken, no vocals",
    "lofi grief waves, sparse piano notes, bass, no drums, 55 BPM, A minor, vast reverb, emotional, no vocals",
    "lofi 4 AM, exhausted piano, bass, no drums, 60 BPM, C minor, heavy reverb, isolated, no vocals",
    "bittersweet lofi, piano melody over minor chords, bass, gentle brushes, 68 BPM, G minor, wide reverb, conflicted, no vocals",
    "lofi yearning, piano ascending lines, bass, sparse kit, 70 BPM, E minor, warm reverb, longing, no vocals",
    "lofi heartbreak, slow piano, bass drone, no drums, 56 BPM, D minor, cathedral reverb, no vocals",
    "lofi missing someone, gentle piano, muted bass, very sparse kit, 65 BPM, A minor, wide reverb, lonely, no vocals",
    "tender lofi, softly played piano, bass, no drums, 62 BPM, G minor, intimate reverb, vulnerable, no vocals",
    "lofi sunset end, descending piano, bass, fade percussion, 64 BPM, C minor, warm reverb, ending, no vocals",
    "lofi what could have been, reflective piano, bass, brushes, 68 BPM, F minor, moderate reverb, regret, no vocals",
    "lofi empty house, solo piano, bass, 58 BPM, D minor, huge reverb, silence between notes, no vocals",
    "lofi letter, piano like typewriter, bass, 70 BPM, A minor, warm reverb, written memories, no vocals",
    "lofi midnight cry, piano in tears, bass drone, 56 BPM, E minor, vast reverb, cathartic, no vocals",
    "lofi holding on, piano with grip, bass, soft kit, 72 BPM, B minor, wide reverb, hopeful melancholy, no vocals",
    "lofi departure, piano sees off, bass, very sparse drums, 65 BPM, G minor, cathedral reverb, farewell, no vocals",
    "lofi growing up, nostalgic piano, bass, gentle shuffle, 74 BPM, C minor, warm reverb, bittersweet, no vocals",
    "lofi what remains, piano echoes, bass, no drums, 60 BPM, F minor, heavy reverb, aftermath, no vocals",
    "lofi winter inside, cold piano, bass, sparse brushes, 65 BPM, A minor, icy reverb, isolated, no vocals",
    "lofi the long way home, tired piano, bass, soft kit, 70 BPM, D minor, warm reverb, journeying, no vocals",
    "lofi letting go, gentle piano resolution, bass, 60 BPM, C major, warm reverb, release, hopeful, no vocals",
    # Warm / cozy (156–170)
    "cozy lofi, warm Rhodes piano, bass, soft kit, fireplace texture, 76 BPM, C major, room reverb, comfortable, no vocals",
    "lofi blanket, soft felt piano, bass, gentle shuffle, 72 BPM, G major, warm reverb, wrapped up, no vocals",
    "lofi hot chocolate, cheerful piano, bass, light drums, 78 BPM, D major, warm analog, cozy kitchen, no vocals",
    "lofi weekend morning, relaxed piano, bass, soft kit, 74 BPM, A major, warm room reverb, at ease, no vocals",
    "lofi home studio, piano with slight saturation, bass, drum groove, 80 BPM, minor key, cassette warmth, creative, no vocals",
    "lofi kitchen vibes, upbeat piano, bass, light drums, 82 BPM, C major, warm reverb, domestic joy, no vocals",
    "lofi sofa session, comfortable piano, bass, gentle groove, 75 BPM, G major, moderate reverb, relaxed, no vocals",
    "lofi cat nap, soft slow piano, bass, no drums, 60 BPM, D major, warm reverb, drowsy, no vocals",
    "lofi afternoon tea, elegant piano, bass, brushes, 76 BPM, A minor, warm reverb, refined comfort, no vocals",
    "lofi reading nook, quiet piano, bass, sparse drums, 70 BPM, F major, room reverb, book in hand, no vocals",
    "lofi Sunday slow, lazy piano, bass, no drums, 65 BPM, C major, warm reverb, nothing to do, no vocals",
    "lofi golden lamp, warm piano, bass, soft kit, 74 BPM, G major, amber reverb, evening comfort, no vocals",
    "lofi knit sweater, cozy piano, bass, gentle shuffle, 72 BPM, A major, warm reverb, November comfort, no vocals",
    "lofi fireplace, Rhodes piano, bass, soft kit, 76 BPM, D minor, warm room reverb, crackling fire, no vocals",
    "lofi hug, gentle piano, bass, soft percussion, 68 BPM, C major, warm reverb, embrace, no vocals",
    # Cosmic / space (171–180)
    "lofi outer space, slow synth pads, piano echoes, sub-bass, no drums, 52 BPM, E minor, vast reverb, cosmic, no vocals",
    "lofi cosmos, ambient pads, Rhodes accents, bass drone, 55 BPM, A minor, cathedral reverb, floating in space, no vocals",
    "deep space lofi, evolving textures, piano, no drums, 50 BPM, C minor, infinite reverb, orbital, no vocals",
    "lofi nebula, swirling pads, gentle piano, bass, 58 BPM, G minor, enormous reverb, mysterious, no vocals",
    "lofi telescope, slow piano under stars, pad swells, bass, no drums, 55 BPM, D minor, cavernous reverb, stargazing, no vocals",
    "lofi lunar, piano on the moon, ambient pads, bass, 60 BPM, F minor, heavy reverb, weightless, no vocals",
    "lofi aurora, shimmering pads, piano melody, bass, 58 BPM, E minor, wide reverb, northern lights, no vocals",
    "lofi satellite, distant piano signal, pad textures, deep bass, 52 BPM, A minor, reverb wash, no vocals",
    "lofi wormhole, evolving pad textures, piano, sub-bass, 50 BPM, C minor, infinite reverb, surreal, no vocals",
    "lofi event horizon, slow pads, piano, no drums, 48 BPM, G minor, vast reverb, gravity, no vocals",
    # Morning / golden hour (181–190)
    "lofi golden morning, bright piano, bass, gentle shuffle, 76 BPM, G major, warm reverb, sunrise, no vocals",
    "lofi first light, opening piano chords, bass, soft kit, 74 BPM, D major, moderate reverb, new day, no vocals",
    "lofi 6 AM, barely awake piano, bass, no drums, 65 BPM, C major, room reverb, soft light, no vocals",
    "lofi morning coffee, energizing piano, bass, light drums, 80 BPM, A major, warm reverb, wake up gently, no vocals",
    "lofi breakfast table, cheerful piano, bass, gentle groove, 78 BPM, G major, bright reverb, morning joy, no vocals",
    "lofi window light, warm piano, bass, sparse drums, 72 BPM, E major, warm room reverb, golden hour, no vocals",
    "lofi garden morning, piano outdoors, bass, soft brushes, 75 BPM, D major, natural reverb, birdsong, no vocals",
    "lofi Sunday morning, slow piano, bass, no drums, 65 BPM, C major, warm reverb, peaceful weekend, no vocals",
    "lofi dawn rising, ascending piano, pads, bass, 68 BPM, E major, wide reverb, hopeful, no vocals",
    "lofi honey light, warm piano, bass, gentle shuffle, 74 BPM, G major, golden reverb, amber morning, no vocals",
    # Slowed / reverb variations (191–200)
    "lofi slowed and reverbed, piano at half speed, bass, vast reverb, 55 BPM, D minor, drenched in reverb, melancholy, no vocals",
    "lofi slowed Rhodes, electric piano slowed, bass, no drums, 52 BPM, A minor, cathedral reverb, underwater, no vocals",
    "lofi reverb drenched, piano swimming in reverb, bass drone, 50 BPM, G minor, infinite reverb, floating, no vocals",
    "lofi slowed and sad, slowed piano and Rhodes, bass, 55 BPM, C minor, heavy reverb, emotional, no vocals",
    "lofi drowned piano, piano lost in reverb, bass, 48 BPM, E minor, enormous reverb, submerged, no vocals",
    "lofi nostalgic slowed, vintage piano at lower speed, bass, 52 BPM, D minor, tape reverb, retro slowed, no vocals",
    "lofi reverb cloud, piano in cloud of reverb, pads, bass, 55 BPM, F minor, cloudy reverb, dreamy, no vocals",
    "lofi slow motion, everything slowed, piano, bass, 45 BPM, A minor, vast reverb, time stretched, no vocals",
    "lofi tape drag, piano with tape drag effect, bass, 50 BPM, C minor, heavy reverb, distorted time, no vocals",
    "lofi endless reverb, piano note echoing forever, bass drone, 48 BPM, G minor, infinite reverb tail, meditative, no vocals",
)

# Short atmosphere phrases randomly injected into preset-derived prompts
# when --randomize-prompts is active.
_ATMOSPHERE_TAGS: tuple[str, ...] = (
    "rain on glass",
    "golden hour glow",
    "late-night city sounds",
    "candlelight atmosphere",
    "empty library at midnight",
    "moonlit room",
    "soft lamp light",
    "4 AM feeling",
    "autumn leaves falling",
    "rooftop at dusk",
    "lonely train station",
    "corner table of a cafe",
    "midnight study session",
    "waking up slowly",
    "half-asleep dreaming",
    "journal entry writing",
    "distant memories",
    "first snow of winter",
    "summer night breeze",
    "morning coffee ritual",
    "watching rain through a window",
    "driving alone at night",
    "empty subway platform",
    "old photograph",
    "waiting for dawn",
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Parameters forwarded to the stable-audio CLI for every generation call."""

    model: str
    steps: int
    cfg_scale: float
    device: str | None
    stable_audio_bin: str
    dry_run: bool


@dataclass
class BatchConfig:
    """Parameters that control a batch generation run."""

    extra_prompt: str | None
    negative_prompt: str
    duration: float
    count: int
    seed_start: int
    show_prompts: bool
    randomize_prompts: bool
    use_bank: bool


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _instrument_label(name: str) -> str:
    return _INSTRUMENT_LABELS.get(name, name.replace("_", " "))


def _reverb_tag(reverb: float) -> str:
    if reverb >= 0.65:
        return "drenched in reverb"
    if reverb >= 0.45:
        return "wide room reverb"
    if reverb >= 0.25:
        return "moderate reverb"
    return "dry minimal reverb"


def _vinyl_tag(vinyl_noise: float) -> str | None:
    if vinyl_noise >= 0.12:
        return "heavy vinyl crackle"
    if vinyl_noise >= 0.05:
        return "vinyl warmth and crackle"
    if vinyl_noise >= 0.02:
        return "faint tape hiss"
    return None


def _lowpass_tag(lowpass_hz: float) -> str | None:
    if lowpass_hz <= 4500:
        return "heavily filtered muffled tone"
    if lowpass_hz <= 6200:
        return "warm low-pass filter"
    return None


def _optional_effect_tags(preset: dict, effects: dict) -> list[str]:
    """Return non-None effect / rhythm description tags derived from a preset."""
    swing = float(preset.get("swing", 0.50))
    drum_density = float(preset.get("drum_density", 0.45))
    tape_wow = float(effects.get("tape_wow", 0.04))
    saturation = float(effects.get("saturation", 0.08))
    vinyl_noise = float(effects.get("vinyl_noise", 0.10))
    lowpass_hz = float(effects.get("lowpass_hz", 8000.0))

    candidates = [
        _vinyl_tag(vinyl_noise),
        _lowpass_tag(lowpass_hz),
        "tape wow flutter" if tape_wow >= 0.06 else None,
        "warm tape saturation" if saturation >= 0.05 else None,
        "laid-back swing groove" if swing >= 0.54 else None,
        "no drums, no percussion" if drum_density <= 0.06
        else ("sparse minimal percussion" if drum_density <= 0.10 else None),
    ]
    return [t for t in candidates if t]


def _instrument_lines(instruments: dict) -> list[str]:
    """Return prompt lines describing the instrumentation."""
    return [
        f"{_instrument_label(instruments.get('chords', 'piano'))} chords",
        f"{_instrument_label(instruments.get('melody', 'piano'))} melody",
        f"{_instrument_label(instruments.get('bass', 'muted bass'))} bass",
        _instrument_label(instruments.get("drums", "soft kit")),
    ]


def build_prompt(
    preset: dict,
    extra_prompt: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Translate a lofi-maker preset dict into a stable-audio text prompt.

    When `rng` is provided, 1–2 random atmosphere phrases are injected to
    vary the prompt across multiple runs of the same preset.
    """
    mood = preset.get("mood", "warm, soft, melancholic")
    bpm_range = preset.get("bpm_range", [72, 88])
    bpm = int((bpm_range[0] + bpm_range[1]) / 2)
    scale_bias: str = preset.get("scale_bias", "minor")
    effects: dict = preset.get("effects", {})

    atmosphere: list[str] = []
    if rng is not None:
        n = rng.randint(1, 2)
        atmosphere = list(rng.sample(_ATMOSPHERE_TAGS, min(n, len(_ATMOSPHERE_TAGS))))

    pieces: list[str] = [
        "lofi hip hop instrumental",
        f"{mood} mood",
        f"{bpm} BPM",
        f"{scale_bias} key",
        *_instrument_lines(preset.get("instruments", {})),
        _reverb_tag(float(effects.get("reverb", 0.30))),
        *_optional_effect_tags(preset, effects),
        *atmosphere,
    ]
    if extra_prompt and extra_prompt.strip():
        pieces.append(extra_prompt.strip())
    pieces += ["no vocals", "no lead vocal", "loopable", "gentle dynamics", "clean instrumental mix"]
    return ", ".join(pieces)


def sample_bank_prompt(rng: random.Random) -> str:
    """Return a uniformly random prompt from the 200-prompt bank."""
    return rng.choice(LOFI_PROMPT_BANK)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_preset(yaml_path: Path) -> dict:
    """Load and return a lofi-maker YAML preset as a plain dict."""
    with open(yaml_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_preset_names(presets_dir: Path) -> list[str]:
    """Return sorted list of preset names (YAML stems) found in presets_dir."""
    return sorted(p.stem for p in presets_dir.glob("*.yaml"))


def _find_stable_audio(override: str | None) -> str | None:
    if override:
        return override
    local = _SCRIPT_DIR / ".venv" / "bin" / "stable-audio"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which("stable-audio")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _run_generation(
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    duration: float,
    seed: int,
    config: GenerationConfig,
) -> bool:
    """Invoke the stable-audio CLI for one track. Returns True on success."""
    cmd = [
        config.stable_audio_bin,
        "--model", config.model,
        "-p", prompt,
        "--duration", str(duration),
        "--steps", str(config.steps),
        "--cfg-scale", str(config.cfg_scale),
        "--seed", str(seed),
        "-o", str(output_path),
    ]
    if negative_prompt:
        cmd += ["--negative-prompt", negative_prompt]
    if config.device:
        cmd += ["--device", config.device]

    if config.dry_run:
        print(f"[dry-run] {config.stable_audio_bin} --model {config.model} \\")
        print(f"  -p {prompt!r} \\")
        if negative_prompt:
            print(f"  --negative-prompt {negative_prompt!r} \\")
        print(f"  --duration {duration} --steps {config.steps} --seed {seed} -o {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, text=True, check=False)
    return result.returncode == 0


def _make_manifest_entry(
    preset_name: str,
    prompt: str,
    negative_prompt: str,
    out_file: Path,
    seed: int,
    success: bool,
    cfg: GenerationConfig,
    batch: BatchConfig,
) -> dict:
    return {
        "preset": preset_name,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "output": str(out_file),
        "duration": batch.duration,
        "model": cfg.model,
        "steps": cfg.steps,
        "cfg_scale": cfg.cfg_scale,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": success,
    }


def _generate_batch(
    selected: list[str],
    presets_dir: Path,
    out_dir: Path,
    batch: BatchConfig,
    gen: GenerationConfig,
) -> list[dict]:
    """Run generation for all selected presets; return JSONL manifest entries."""
    entries: list[dict] = []
    total = len(selected) * batch.count
    seed_counter = batch.seed_start
    rng = random.Random(batch.seed_start) if batch.randomize_prompts or batch.use_bank else None

    for idx_p, preset_name in enumerate(selected):
        if batch.use_bank and rng is not None:
            prompt = sample_bank_prompt(rng)
        else:
            preset_data = load_preset(presets_dir / f"{preset_name}.yaml")
            prompt = build_prompt(preset_data, batch.extra_prompt, rng)

        if batch.show_prompts:
            print(f"\n[{preset_name}]")
            print(f"  prompt: {prompt}")
            if batch.negative_prompt:
                print(f"  negative: {batch.negative_prompt}")

        for i in range(batch.count):
            current = idx_p * batch.count + i + 1
            suffix = f"_{i + 1:02d}" if batch.count > 1 else ""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = out_dir / f"{preset_name}{suffix}_{timestamp}.wav"
            current_seed = seed_counter
            seed_counter += 1

            if not gen.dry_run:
                print(f"[{current}/{total}] {preset_name}{suffix}  seed={current_seed}  duration={batch.duration}s")

            ok = _run_generation(prompt, batch.negative_prompt, out_file, batch.duration, current_seed, gen)
            entries.append(_make_manifest_entry(
                preset_name, prompt, batch.negative_prompt, out_file, current_seed, ok, gen, batch,
            ))

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_from_lofi_presets.py",
        description="Generate audio via stable-audio CLI from lofi-maker presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  LOFI_PRESETS_DIR     Override --presets-dir\n"
            "  STABLE_AUDIO_BIN     Override --stable-audio-bin\n"
            "  HF_TOKEN             HuggingFace token for gated model access\n"
        ),
    )

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--preset", metavar="NAME", help="Generate from a single named preset")
    sel.add_argument("--all", action="store_true", help="Generate from every available preset")
    sel.add_argument("--random", action="store_true", help="Pick one preset at random")
    sel.add_argument("--random-from-bank", action="store_true",
                     help="Generate directly from the 200-prompt bank (ignores preset parameters)")
    sel.add_argument("--list", action="store_true", help="List available presets and exit")

    parser.add_argument("--count", "-n", type=int, default=1, metavar="N",
                        help="Tracks per preset (default: 1)")
    parser.add_argument("--duration", "-d", type=float, default=240.0, metavar="SECS",
                        help="Track duration in seconds (default: 240; medium max: 380)")
    parser.add_argument("--model", "-m", default="medium",
                        choices=["small-music", "small-sfx", "medium",
                                 "small-music-base", "small-sfx-base", "medium-base"],
                        help="stable-audio model (default: medium)")
    parser.add_argument("--steps", type=int, default=8, metavar="N",
                        help="Diffusion steps (default: 8; try 20–50 for higher quality)")
    parser.add_argument("--cfg-scale", type=float, default=1.0, metavar="F",
                        help="CFG scale (default: 1.0; use 7.0 for *-base models)")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Base seed; incremented per track. Default: random")
    parser.add_argument("--device", default=None, metavar="DEVICE",
                        help="Torch device: cuda / mps / cpu (auto-detected by default)")

    parser.add_argument("--extra-prompt", metavar="TEXT",
                        help="Extra text appended to every generated prompt")
    parser.add_argument("--negative-prompt", metavar="TEXT", default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt (default: suppresses fast/harsh/vocal elements)")
    parser.add_argument("--no-negative-prompt", action="store_true",
                        help="Disable the negative prompt entirely")
    parser.add_argument("--randomize-prompts", action="store_true",
                        help="Inject 1–2 random atmosphere phrases per track for variety")

    parser.add_argument("--out-dir", "-o", default="output/stable_audio", metavar="DIR",
                        help="Output directory (default: output/stable_audio)")
    parser.add_argument("--manifest", metavar="PATH",
                        help="JSONL manifest path (default: OUT_DIR/manifest.jsonl)")

    parser.add_argument("--presets-dir", metavar="DIR",
                        default=os.environ.get("LOFI_PRESETS_DIR", str(DEFAULT_PRESETS_DIR)),
                        help=f"Preset YAML directory (default: {DEFAULT_PRESETS_DIR})")
    parser.add_argument("--stable-audio-bin", metavar="PATH",
                        default=os.environ.get("STABLE_AUDIO_BIN"),
                        help="Path to stable-audio executable")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts and commands without running generation")
    parser.add_argument("--print-prompts", action="store_true",
                        help="Print the resolved prompt for each preset (then generate)")

    return parser


def _resolve_presets(args: argparse.Namespace, preset_names: list[str]) -> list[str]:
    """Return the list of preset names to generate from CLI selection flags."""
    if args.all or args.random_from_bank:
        return preset_names if args.all else [random.choice(preset_names)]
    if args.random:
        return [random.choice(preset_names)]
    if args.preset not in preset_names:
        print(f"Error: preset {args.preset!r} not found. Run --list to see available presets.",
              file=sys.stderr)
        sys.exit(1)
    return [args.preset]


def main() -> None:
    """Entry point: parse args, validate environment, and run generation."""
    parser = _build_parser()
    args = parser.parse_args()

    presets_dir = Path(args.presets_dir)
    if not presets_dir.is_dir():
        print(f"Error: presets directory not found: {presets_dir}\n"
              f"  Set --presets-dir or LOFI_PRESETS_DIR.", file=sys.stderr)
        sys.exit(1)

    preset_names = list_preset_names(presets_dir)
    if not preset_names:
        print(f"Error: no .yaml presets found in {presets_dir}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print(f"Presets in {presets_dir}:")
        for name in preset_names:
            print(f"  {name}")
        return

    if not any([args.preset, args.all, args.random, args.random_from_bank]):
        parser.print_usage()
        print("\nSpecify one of: --preset NAME  --all  --random  --random-from-bank  --list",
              file=sys.stderr)
        sys.exit(1)

    stable_audio = _find_stable_audio(args.stable_audio_bin)
    if stable_audio is None and not args.dry_run:
        print(
            "Error: stable-audio executable not found. Run 'uv sync' in stable-audio-3/.\n"
            "  Also ensure HuggingFace access and login:  huggingface-cli login",
            file=sys.stderr,
        )
        sys.exit(1)

    selected = _resolve_presets(args, preset_names)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.jsonl"
    seed_start = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    batch = BatchConfig(
        extra_prompt=args.extra_prompt,
        negative_prompt="" if args.no_negative_prompt else args.negative_prompt,
        duration=args.duration,
        count=args.count,
        seed_start=seed_start,
        show_prompts=args.print_prompts or args.dry_run,
        randomize_prompts=args.randomize_prompts,
        use_bank=args.random_from_bank,
    )
    gen = GenerationConfig(
        model=args.model,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=args.device,
        stable_audio_bin=stable_audio or "stable-audio",
        dry_run=args.dry_run,
    )

    entries = _generate_batch(selected, presets_dir, out_dir, batch, gen)

    if args.dry_run:
        print(f"\n[dry-run] Would generate {len(entries)} track(s) to {out_dir}/")
        return

    with open(manifest_path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    success_count = sum(1 for e in entries if e["success"])
    print(f"\nDone: {success_count}/{len(entries)} tracks generated.")
    print(f"Output: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
