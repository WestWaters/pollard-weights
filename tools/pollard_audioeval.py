#!/usr/bin/env python3
"""pollard-audioeval -- did quantization break the model's VOICE or its EARS, and in which language?

pollard-mmeval asks whether a rung can still see; this asks whether it can still speak and listen. Every other
number on a Pollard card is text: perplexity, KL divergence, top-1 -- English-heavy text at that. A speech model can
read fine on all of them while its TTS slurs in Japanese, because the backbone reads back its own audio codes when it
speaks and no text calibration ever covers those activations (LFM3-Audio IQ4_XS, reported 2026-10-29: "Q6 totally
fine, IQ4 deterioration in TTS, especially in languages that are not English").

    pollard-audioeval --gguf rung.gguf --mmproj mmproj-BF16.gguf --judge asr-f16.gguf --judge-mmproj asr-mm.gguf
    pollard-audioeval --gguf rung.gguf --mmproj mm.gguf --ref q6_k.gguf --judge ... --judge-mmproj ... --out r.json

Two probes, both per language:
  speech out  the rung speaks fixed sentences (llama-tts, fixed seed -- byte-identical across runs), a FIXED judge ASR
              (default role: an f16 multilingual ASR model, never the model under test) transcribes them; character
              error vs the sentence, plus the |log| duration ratio vs the reference rung's audio (silence, runaway)
  speech in   the reference rung speaks the sentences, the rung under test transcribes them (its own ASR prompt);
              character error vs the sentence
  long        the rung reads an eight-sentence passage in ~10 s chunks, the way apps drive TTS: character error over
              the passage, duration drift, and the repetition it adds (Moshi: 3-bit artifacts grow with length)

Read per language and by the WORST language, not the mean: the failure Lucas saw is invisible in an English mean.
Report-only, like every Pollard number: it goes on the card, it never blocks a rung.

What it does NOT measure: voice quality / speaker similarity (no MOS model here), spoken tool calls, long-form
stability. A rung that passes here still might sound worse; one that fails here is broken for that language.

Imports nothing from the rest of Pollard. Owns its sentences, subprocess handling and scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave

#: llama-tts --tts-lang code -> (English name for the ASR prompts, probe sentences). Plain everyday sentences:
#: numbers spelled out, no names that a judge may spell two ways. Escaped so the source stays ASCII (a cp1252
#: console must never meet a tool's bytes); the text itself is unchanged.
SENTENCES = {
    "en": ("English", [
        'The weather tomorrow will be cloudy with rain in the afternoon.',
        'Please bring two bottles of water and a warm jacket to the meeting.',
        'After the train was delayed for almost an hour, we decided to walk the rest of the way along the river.',
        'My grandmother keeps a small garden behind her house, where she grows tomatoes, beans and fresh herbs every summer.',
    ]),
    "de": ("German", [
        'Das Wetter morgen wird bew\xf6lkt sein, am Nachmittag regnet es.',
        'Bitte bring zwei Flaschen Wasser und eine warme Jacke zum Treffen mit.',
        'Nachdem der Zug fast eine Stunde Versp\xe4tung hatte, beschlossen wir, den Rest des Weges am Fluss entlang zu laufen.',
        'Meine Gro\xdfmutter hat einen kleinen Garten hinter ihrem Haus, wo sie jeden Sommer Tomaten, Bohnen und frische Kr\xe4uter anbaut.',
    ]),
    "es": ("Spanish", [
        'Ma\xf1ana el tiempo estar\xe1 nublado y llover\xe1 por la tarde.',
        'Por favor, trae dos botellas de agua y una chaqueta abrigada a la reuni\xf3n.',
        'Despu\xe9s de que el tren se retrasara casi una hora, decidimos caminar el resto del camino junto al r\xedo.',
        'Mi abuela tiene un peque\xf1o jard\xedn detr\xe1s de su casa, donde cultiva tomates, jud\xedas y hierbas frescas cada verano.',
    ]),
    "fr": ("French", [
        "Demain, le temps sera nuageux avec de la pluie dans l'apr\xe8s-midi.",
        "Merci d'apporter deux bouteilles d'eau et une veste chaude \xe0 la r\xe9union.",
        'Apr\xe8s que le train a eu presque une heure de retard, nous avons d\xe9cid\xe9 de faire le reste du chemin \xe0 pied le long de la rivi\xe8re.',
        'Ma grand-m\xe8re a un petit jardin derri\xe8re sa maison, o\xf9 elle cultive des tomates, des haricots et des herbes fra\xeeches chaque \xe9t\xe9.',
    ]),
    "it": ("Italian", [
        'Domani il tempo sar\xe0 nuvoloso con pioggia nel pomeriggio.',
        "Per favore, porta due bottiglie d'acqua e una giacca pesante alla riunione.",
        "Dopo che il treno era in ritardo di quasi un'ora, abbiamo deciso di fare il resto della strada a piedi lungo il fiume.",
        'Mia nonna ha un piccolo orto dietro casa, dove ogni estate coltiva pomodori, fagioli ed erbe fresche.',
    ]),
    "pt": ("Portuguese", [
        'Amanh\xe3 o tempo estar\xe1 nublado, com chuva \xe0 tarde.',
        'Por favor, traga duas garrafas de \xe1gua e um casaco quente para a reuni\xe3o.',
        'Depois de o comboio se atrasar quase uma hora, decidimos fazer o resto do caminho a p\xe9 ao longo do rio.',
        'A minha av\xf3 tem uma pequena horta atr\xe1s de casa, onde cultiva tomates, feij\xf5es e ervas frescas todos os ver\xf5es.',
    ]),
    "nl": ("Dutch", [
        'Het weer wordt morgen bewolkt met regen in de middag.',
        'Neem alsjeblieft twee flessen water en een warme jas mee naar de vergadering.',
        'Nadat de trein bijna een uur vertraging had, besloten we de rest van de weg langs de rivier te lopen.',
        'Mijn oma heeft een kleine tuin achter haar huis, waar ze elke zomer tomaten, bonen en verse kruiden kweekt.',
    ]),
    "sv": ("Swedish", [
        'V\xe4dret i morgon blir molnigt med regn p\xe5 eftermiddagen.',
        'Ta g\xe4rna med tv\xe5 flaskor vatten och en varm jacka till m\xf6tet.',
        'Efter att t\xe5get hade blivit n\xe4stan en timme f\xf6rsenat best\xe4mde vi oss f\xf6r att g\xe5 resten av v\xe4gen l\xe4ngs floden.',
        'Min mormor har en liten tr\xe4dg\xe5rd bakom sitt hus d\xe4r hon odlar tomater, b\xf6nor och f\xe4rska \xf6rter varje sommar.',
    ]),
    "da": ("Danish", [
        'Vejret i morgen bliver overskyet med regn om eftermiddagen.',
        'Tag venligst to flasker vand og en varm jakke med til m\xf8det.',
        'Efter at toget var n\xe6sten en time forsinket, besluttede vi at g\xe5 resten af vejen langs floden.',
        'Min mormor har en lille have bag sit hus, hvor hun dyrker tomater, b\xf8nner og friske krydderurter hver sommer.',
    ]),
    "fi": ("Finnish", [
        'Huomenna on pilvist\xe4 ja iltap\xe4iv\xe4ll\xe4 sataa.',
        'Ota kokoukseen mukaan kaksi pulloa vett\xe4 ja l\xe4mmin takki.',
        'Kun juna oli l\xe4hes tunnin my\xf6h\xe4ss\xe4, p\xe4\xe4timme k\xe4vell\xe4 loppumatkan joen vartta pitkin.',
        'Iso\xe4idill\xe4ni on pieni puutarha talonsa takana, ja siell\xe4 h\xe4n kasvattaa joka kes\xe4 tomaatteja, papuja ja tuoreita yrttej\xe4.',
    ]),
    "pl": ("Polish", [
        'Jutro b\u0119dzie pochmurno, a po po\u0142udniu spadnie deszcz.',
        'Prosz\u0119 przynie\u015b\u0107 na spotkanie dwie butelki wody i ciep\u0142\u0105 kurtk\u0119.',
        'Poniewa\u017c poci\u0105g sp\xf3\u017ani\u0142 si\u0119 prawie godzin\u0119, postanowili\u015bmy przej\u015b\u0107 reszt\u0119 drogi pieszo wzd\u0142u\u017c rzeki.',
        'Moja babcia ma ma\u0142y ogr\xf3dek za domem, gdzie ka\u017cdego lata uprawia pomidory, fasol\u0119 i \u015bwie\u017ce zio\u0142a.',
    ]),
    "tr": ("Turkish", [
        'Yar\u0131n hava bulutlu olacak ve \xf6\u011fleden sonra ya\u011fmur ya\u011facak.',
        'L\xfctfen toplant\u0131ya iki \u015fi\u015fe su ve kal\u0131n bir ceket getir.',
        'Tren neredeyse bir saat geciktikten sonra yolun geri kalan\u0131n\u0131 nehir boyunca y\xfcr\xfcmeye karar verdik.',
        'B\xfcy\xfckannemin evinin arkas\u0131nda k\xfc\xe7\xfck bir bah\xe7esi var ve her yaz orada domates, fasulye ve taze otlar yeti\u015ftiriyor.',
    ]),
    "el": ("Greek", [
        '\u0391\u03cd\u03c1\u03b9\u03bf \u03bf \u03ba\u03b1\u03b9\u03c1\u03cc\u03c2 \u03b8\u03b1 \u03b5\u03af\u03bd\u03b1\u03b9 \u03c3\u03c5\u03bd\u03bd\u03b5\u03c6\u03b9\u03b1\u03c3\u03bc\u03ad\u03bd\u03bf\u03c2 \u03bc\u03b5 \u03b2\u03c1\u03bf\u03c7\u03ae \u03c4\u03bf \u03b1\u03c0\u03cc\u03b3\u03b5\u03c5\u03bc\u03b1.',
        '\u03a0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb\u03ce \u03c6\u03ad\u03c1\u03b5 \u03b4\u03cd\u03bf \u03bc\u03c0\u03bf\u03c5\u03ba\u03ac\u03bb\u03b9\u03b1 \u03bd\u03b5\u03c1\u03cc \u03ba\u03b1\u03b9 \u03ad\u03bd\u03b1 \u03b6\u03b5\u03c3\u03c4\u03cc \u03bc\u03c0\u03bf\u03c5\u03c6\u03ac\u03bd \u03c3\u03c4\u03b7 \u03c3\u03c5\u03bd\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7.',
        '\u0391\u03c6\u03bf\u03cd \u03c4\u03bf \u03c4\u03c1\u03ad\u03bd\u03bf \u03ba\u03b1\u03b8\u03c5\u03c3\u03c4\u03ad\u03c1\u03b7\u03c3\u03b5 \u03c3\u03c7\u03b5\u03b4\u03cc\u03bd \u03bc\u03af\u03b1 \u03ce\u03c1\u03b1, \u03b1\u03c0\u03bf\u03c6\u03b1\u03c3\u03af\u03c3\u03b1\u03bc\u03b5 \u03bd\u03b1 \u03c0\u03b5\u03c1\u03c0\u03b1\u03c4\u03ae\u03c3\u03bf\u03c5\u03bc\u03b5 \u03c4\u03b7\u03bd \u03c5\u03c0\u03cc\u03bb\u03bf\u03b9\u03c0\u03b7 \u03b4\u03b9\u03b1\u03b4\u03c1\u03bf\u03bc\u03ae \u03b4\u03af\u03c0\u03bb\u03b1 \u03c3\u03c4\u03bf \u03c0\u03bf\u03c4\u03ac\u03bc\u03b9.',
        '\u0397 \u03b3\u03b9\u03b1\u03b3\u03b9\u03ac \u03bc\u03bf\u03c5 \u03ad\u03c7\u03b5\u03b9 \u03ad\u03bd\u03b1\u03bd \u03bc\u03b9\u03ba\u03c1\u03cc \u03ba\u03ae\u03c0\u03bf \u03c0\u03af\u03c3\u03c9 \u03b1\u03c0\u03cc \u03c4\u03bf \u03c3\u03c0\u03af\u03c4\u03b9 \u03c4\u03b7\u03c2, \u03cc\u03c0\u03bf\u03c5 \u03ba\u03ac\u03b8\u03b5 \u03ba\u03b1\u03bb\u03bf\u03ba\u03b1\u03af\u03c1\u03b9 \u03ba\u03b1\u03bb\u03bb\u03b9\u03b5\u03c1\u03b3\u03b5\u03af \u03bd\u03c4\u03bf\u03bc\u03ac\u03c4\u03b5\u03c2, \u03c6\u03b1\u03c3\u03cc\u03bb\u03b9\u03b1 \u03ba\u03b1\u03b9 \u03c6\u03c1\u03ad\u03c3\u03ba\u03b1 \u03bc\u03c5\u03c1\u03c9\u03b4\u03b9\u03ba\u03ac.',
    ]),
    "ar": ("Arabic", [
        '\u0633\u064a\u0643\u0648\u0646 \u0627\u0644\u0637\u0642\u0633 \u063a\u0627\u0626\u0645\u0627 \u063a\u062f\u0627 \u0645\u0639 \u0623\u0645\u0637\u0627\u0631 \u0641\u064a \u0641\u062a\u0631\u0629 \u0628\u0639\u062f \u0627\u0644\u0638\u0647\u0631.',
        '\u0645\u0646 \u0641\u0636\u0644\u0643 \u0623\u062d\u0636\u0631 \u0632\u062c\u0627\u062c\u062a\u064a\u0646 \u0645\u0646 \u0627\u0644\u0645\u0627\u0621 \u0648\u0633\u062a\u0631\u0629 \u062f\u0627\u0641\u0626\u0629 \u0625\u0644\u0649 \u0627\u0644\u0627\u062c\u062a\u0645\u0627\u0639.',
        '\u0628\u0639\u062f \u0623\u0646 \u062a\u0623\u062e\u0631 \u0627\u0644\u0642\u0637\u0627\u0631 \u0646\u062d\u0648 \u0633\u0627\u0639\u0629\u060c \u0642\u0631\u0631\u0646\u0627 \u0623\u0646 \u0646\u0645\u0634\u064a \u0628\u0642\u064a\u0629 \u0627\u0644\u0637\u0631\u064a\u0642 \u0639\u0644\u0649 \u0636\u0641\u0629 \u0627\u0644\u0646\u0647\u0631.',
        '\u0644\u062f\u0649 \u062c\u062f\u062a\u064a \u062d\u062f\u064a\u0642\u0629 \u0635\u063a\u064a\u0631\u0629 \u062e\u0644\u0641 \u0645\u0646\u0632\u0644\u0647\u0627\u060c \u062a\u0632\u0631\u0639 \u0641\u064a\u0647\u0627 \u0627\u0644\u0637\u0645\u0627\u0637\u0645 \u0648\u0627\u0644\u0641\u0627\u0635\u0648\u0644\u064a\u0627 \u0648\u0627\u0644\u0623\u0639\u0634\u0627\u0628 \u0627\u0644\u0637\u0627\u0632\u062c\u0629 \u0643\u0644 \u0635\u064a\u0641.',
    ]),
    "he": ("Hebrew", [
        '\u05de\u05d7\u05e8 \u05de\u05d6\u05d2 \u05d4\u05d0\u05d5\u05d5\u05d9\u05e8 \u05d9\u05d4\u05d9\u05d4 \u05de\u05e2\u05d5\u05e0\u05df \u05e2\u05dd \u05d2\u05e9\u05dd \u05d0\u05d7\u05e8 \u05d4\u05e6\u05d4\u05e8\u05d9\u05d9\u05dd.',
        '\u05d1\u05d1\u05e7\u05e9\u05d4 \u05ea\u05d1\u05d9\u05d0 \u05e9\u05e0\u05d9 \u05d1\u05e7\u05d1\u05d5\u05e7\u05d9 \u05de\u05d9\u05dd \u05d5\u05de\u05e2\u05d9\u05dc \u05d7\u05dd \u05dc\u05e4\u05d2\u05d9\u05e9\u05d4.',
        '\u05d0\u05d7\u05e8\u05d9 \u05e9\u05d4\u05e8\u05db\u05d1\u05ea \u05d4\u05ea\u05e2\u05db\u05d1\u05d4 \u05db\u05de\u05e2\u05d8 \u05e9\u05e2\u05d4, \u05d4\u05d7\u05dc\u05d8\u05e0\u05d5 \u05dc\u05dc\u05db\u05ea \u05d0\u05ea \u05e9\u05d0\u05e8 \u05d4\u05d3\u05e8\u05da \u05dc\u05d0\u05d5\u05e8\u05da \u05d4\u05e0\u05d4\u05e8.',
        '\u05dc\u05e1\u05d1\u05ea\u05d0 \u05e9\u05dc\u05d9 \u05d9\u05e9 \u05d2\u05d9\u05e0\u05d4 \u05e7\u05d8\u05e0\u05d4 \u05de\u05d0\u05d7\u05d5\u05e8\u05d9 \u05d4\u05d1\u05d9\u05ea, \u05d5\u05d1\u05db\u05dc \u05e7\u05d9\u05e5 \u05d4\u05d9\u05d0 \u05de\u05d2\u05d3\u05dc\u05ea \u05d1\u05d4 \u05e2\u05d2\u05d1\u05e0\u05d9\u05d5\u05ea, \u05e9\u05e2\u05d5\u05e2\u05d9\u05ea \u05d5\u05e2\u05e9\u05d1\u05d9 \u05ea\u05d9\u05d1\u05d5\u05dc \u05d8\u05e8\u05d9\u05d9\u05dd.',
    ]),
    "ja": ("Japanese", [
        '\u660e\u65e5\u306e\u5929\u6c17\u306f\u66c7\u308a\u3067\u3001\u5348\u5f8c\u306f\u96e8\u304c\u964d\u308b\u3067\u3057\u3087\u3046\u3002',
        '\u4f1a\u8b70\u306b\u306f\u6c34\u3092\u4e8c\u672c\u3068\u6696\u304b\u3044\u4e0a\u7740\u3092\u6301\u3063\u3066\u304d\u3066\u304f\u3060\u3055\u3044\u3002',
        '\u96fb\u8eca\u304c\u4e00\u6642\u9593\u8fd1\u304f\u9045\u308c\u305f\u306e\u3067\u3001\u6b8b\u308a\u306e\u9053\u306f\u5ddd\u6cbf\u3044\u3092\u6b69\u304f\u3053\u3068\u306b\u3057\u307e\u3057\u305f\u3002',
        '\u7956\u6bcd\u306f\u5bb6\u306e\u88cf\u306b\u5c0f\u3055\u306a\u7551\u3092\u6301\u3063\u3066\u3044\u3066\u3001\u6bce\u5e74\u590f\u306b\u306a\u308b\u3068\u30c8\u30de\u30c8\u3084\u8c46\u3001\u65b0\u9bae\u306a\u30cf\u30fc\u30d6\u3092\u80b2\u3066\u3066\u3044\u307e\u3059\u3002',
    ]),
    "ko": ("Korean", [
        '\ub0b4\uc77c \ub0a0\uc528\ub294 \ud750\ub9ac\uace0 \uc624\ud6c4\uc5d0\ub294 \ube44\uac00 \uc624\uaca0\uc2b5\ub2c8\ub2e4.',
        '\ud68c\uc758\uc5d0 \ubb3c \ub450 \ubcd1\uacfc \ub530\ub73b\ud55c \uc7ac\ud0b7\uc744 \uac00\uc838\uc640 \uc8fc\uc138\uc694.',
        '\uae30\ucc28\uac00 \uac70\uc758 \ud55c \uc2dc\uac04\uc774\ub098 \ub2a6\uc5b4\uc838\uc11c \uc6b0\ub9ac\ub294 \ub098\uba38\uc9c0 \uae38\uc744 \uac15\uc744 \ub530\ub77c \uac77\uae30\ub85c \ud588\uc2b5\ub2c8\ub2e4.',
        '\ud560\uba38\ub2c8\ub294 \uc9d1 \ub4a4\uc5d0 \uc791\uc740 \ud143\ubc2d\uc774 \uc788\uc5b4\uc11c \ub9e4\ub144 \uc5ec\ub984 \ud1a0\ub9c8\ud1a0\uc640 \ucf69, \uc2e0\uc120\ud55c \ud5c8\ube0c\ub97c \uae30\ub974\uc2ed\ub2c8\ub2e4.',
    ]),
    "zh": ("Chinese", [
        '\u660e\u5929\u5929\u6c14\u591a\u4e91\uff0c\u4e0b\u5348\u6709\u96e8\u3002',
        '\u8bf7\u5e26\u4e24\u74f6\u6c34\u548c\u4e00\u4ef6\u6696\u548c\u7684\u5916\u5957\u6765\u5f00\u4f1a\u3002',
        '\u706b\u8f66\u665a\u70b9\u4e86\u5c06\u8fd1\u4e00\u4e2a\u5c0f\u65f6\uff0c\u6240\u4ee5\u6211\u4eec\u51b3\u5b9a\u6cbf\u7740\u6cb3\u8fb9\u8d70\u5b8c\u5269\u4e0b\u7684\u8def\u3002',
        '\u6211\u5976\u5976\u5728\u623f\u5b50\u540e\u9762\u6709\u4e00\u4e2a\u5c0f\u83dc\u56ed\uff0c\u6bcf\u5e74\u590f\u5929\u5979\u90fd\u5728\u90a3\u91cc\u79cd\u897f\u7ea2\u67ff\u3001\u8c46\u5b50\u548c\u65b0\u9c9c\u7684\u9999\u8349\u3002',
    ]),
}
#: no spaces between words: score on characters with whitespace dropped
_NO_SPACE = {"ja", "zh"}
SEED = 7


# ------------------------------------------------------------------------------------------------ scoring

def normalize(text, lang):
    """NFKC, lower case, punctuation and symbols dropped, whitespace collapsed (removed for ja/zh)."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = "".join(" " if unicodedata.category(ch)[0] in "PSZ" else ch for ch in t)
    t = " ".join(t.split())
    return t.replace(" ", "") if lang in _NO_SPACE else t


def cer(ref, hyp, lang):
    """Character error rate (edit distance / reference length) after normalize(); capped at 1.0 so one runaway
    transcript can't swamp a language's mean."""
    a, b = normalize(ref, lang), normalize(hyp, lang)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return min(1.0, prev[-1] / len(a))


def clean_transcript(out, prompt=None):
    """The transcript in a CLI's output: the text after an `<asr_text>` tag when there is one (Qwen3-ASR prints
    `language X<asr_text>...`), else the last non-empty line; a prompt echoed back is removed."""
    text = out.strip()
    if "<asr_text>" in text:
        text = text.rsplit("<asr_text>", 1)[1]
    else:
        lines = [l for l in text.splitlines() if l.strip()]
        text = lines[-1] if lines else ""
    if prompt and text.endswith(prompt):
        text = text[: -len(prompt)]
    return re.sub(r"<[^>]{1,40}>", "", text).strip()


def repetition(text, lang, n=4):
    """Share of character n-grams (word n-grams for spaced scripts) that repeat an earlier one: a looping or
    stuttering rung (Moshi's long-generation failure) drives it up; normal speech of these passages stays low."""
    t = normalize(text, lang)
    units = list(t) if lang in _NO_SPACE else t.split()
    n = n if lang in _NO_SPACE else 3
    grams = [tuple(units[i:i + n]) for i in range(max(0, len(units) - n + 1))]
    return round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0


def wav_seconds(path):
    try:
        with wave.open(path) as w:
            return w.getnframes() / float(w.getframerate())
    except (OSError, wave.Error, EOFError):
        return 0.0


# ------------------------------------------------------------------------------------------------ runtime

def find_bin(name, hint=None):
    for d in (hint, os.environ.get("POLLARD_LLAMA_BIN"),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "llama.cpp",
                           "build", "bin")):
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    p = shutil.which(name)
    if not p:
        raise SystemExit(f"{name} not found: pass --bin or set POLLARD_LLAMA_BIN")
    return p


def speak(bin_dir, model, mmproj, text, lang, out, ngl, voice="female", timeout=600, seed=None):
    cmd = [find_bin("llama-tts", bin_dir), "-m", model, "-mm", mmproj, "-p", text, "--tts-lang", lang,
           "--tts-voice", voice, "--seed", str(SEED if seed is None else seed), "-ngl", str(ngl), "--output", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 44


def transcribe(bin_dir, model, mmproj, audio, ngl, system=None, prompt="Transcribe the audio.", timeout=600, n=160):
    cmd = [find_bin("llama-mtmd-cli", bin_dir), "-m", model, "--mmproj", mmproj, "--audio", audio, "-p", prompt,
           "-n", str(n), "--temp", "0", "-ngl", str(ngl)]
    if system:
        cmd += ["-sys", system]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return clean_transcript(r.stdout, prompt) if r.returncode == 0 else ""


# ------------------------------------------------------------------------------------------------ probes

def run(a):
    langs = [l for l in (a.langs.split(",") if a.langs else SENTENCES) if l in SENTENCES]
    ref, ref_mm = a.ref or a.gguf, a.ref_mmproj or a.mmproj
    res = {"model": a.gguf, "mmproj": a.mmproj, "ref": ref, "judge": a.judge,
           "seed": SEED if getattr(a, "seed", None) is None else a.seed, "langs": {}}
    work = a.keep or tempfile.mkdtemp(prefix="audioeval-")
    for lang in langs:
        name, sents = SENTENCES[lang]
        row = {"speak_cer": [], "speak_dur_drift": [], "listen_cer": []}
        for i, s in enumerate(sents):
            mine, theirs = os.path.join(work, f"{lang}{i}-m.wav"), os.path.join(work, f"{lang}{i}-r.wav")
            if "out" in a.probes:
                ok = speak(a.bin, a.gguf, a.mmproj, s, lang, mine, a.ngl, seed=getattr(a, "seed", None))
                heard = transcribe(a.bin, a.judge, a.judge_mmproj, mine, a.ngl) if ok else ""
                row["speak_cer"].append(round(cer(s, heard, lang), 4))
                row.setdefault("speak_heard", []).append(heard)
            if ("in" in a.probes or ("out" in a.probes and a.ref)) and not os.path.isfile(theirs):
                speak(a.bin, ref, ref_mm, s, lang, theirs, a.ngl)
            if "out" in a.probes and a.ref and os.path.isfile(mine) and os.path.isfile(theirs):
                dm, dr = wav_seconds(mine), wav_seconds(theirs)
                row["speak_dur_drift"].append(round(abs(math.log(dm / dr)), 4) if dm and dr else 1.0)
            if "in" in a.probes:
                heard = transcribe(a.bin, a.gguf, a.mmproj, theirs, a.ngl, system=f"Perform ASR in {name}.") \
                    if os.path.isfile(theirs) else ""
                row["listen_cer"].append(round(cer(s, heard, lang), 4))
                row.setdefault("listen_heard", []).append(heard)
        if "long" in a.probes:
            # A long passage read the way apps read one: in chunks of about two sentences (~10 s each), every
            # chunk spoken and transcribed on its own, scored over the whole passage. One 30-60 s generation is
            # not a usable probe -- LFM3-Audio's own f16 skips and loops on it -- but chunks of this size are how
            # it is actually driven, and they still run several times longer than the single-sentence probe.
            sep = "" if lang in _NO_SPACE else " "
            seq = sents + sents[::-1]
            chunks = [sep.join(seq[i:i + 2]) for i in range(0, len(seq), 2)]
            heard, dm, dr = [], 0.0, 0.0
            for c_i, chunk in enumerate(chunks):
                mine, theirs = os.path.join(work, f"{lang}L{c_i}-m.wav"), os.path.join(work, f"{lang}L{c_i}-r.wav")
                ok = speak(a.bin, a.gguf, a.mmproj, chunk, lang, mine, a.ngl, seed=getattr(a, "seed", None))
                heard.append(transcribe(a.bin, a.judge, a.judge_mmproj, mine, a.ngl, n=400) if ok else "")
                if a.ref:
                    if not os.path.isfile(theirs):
                        speak(a.bin, ref, ref_mm, chunk, lang, theirs, a.ngl)
                    dm, dr = dm + wav_seconds(mine), dr + wav_seconds(theirs)
            passage, said = sep.join(chunks), sep.join(heard)
            row["long_cer"] = round(cer(passage, said, lang), 4)
            row["long_repeat"] = round(max(0.0, repetition(said, lang) - repetition(passage, lang)), 4)
            if a.ref:
                row["long_dur_drift"] = round(abs(math.log(dm / dr)), 4) if dm and dr else 1.0
            row["long_heard"] = said
        for k in ("speak_cer", "speak_dur_drift", "listen_cer"):
            row[k + "_mean"] = round(sum(row[k]) / len(row[k]), 4) if row[k] else None
        res["langs"][lang] = row
        print(f"  {lang:3s} speak CER {row['speak_cer_mean']}  drift {row['speak_dur_drift_mean']}  "
              f"listen CER {row['listen_cer_mean']}", flush=True)
    for k in ("speak_cer_mean", "listen_cer_mean", "speak_dur_drift_mean", "long_cer", "long_repeat",
              "long_dur_drift"):
        vals = {l: r[k] for l, r in res["langs"].items() if r.get(k) is not None}
        if vals:
            worst = max(vals, key=vals.get)
            res[k.replace("_mean", "")] = {"mean": round(sum(vals.values()) / len(vals), 4),
                                           "worst": vals[worst], "worst_lang": worst}
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard-audioeval", description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="the rung under test")
    ap.add_argument("--mmproj", required=True, help="its mmproj (audio encoder + speech output)")
    ap.add_argument("--ref", help="reference rung (Q6_K / f16) for duration drift and the speech-in clips")
    ap.add_argument("--ref-mmproj", help="the reference's mmproj (default: --mmproj, so only the backbone differs)")
    ap.add_argument("--judge", required=True, help="fixed ASR model that scores speech out (f16, never the rung)")
    ap.add_argument("--judge-mmproj", required=True)
    ap.add_argument("--langs", help="comma list of language codes (default: every language)")
    ap.add_argument("--probes", default="out,in", help="out (TTS), in (ASR), long (a chunked passage), comma list")
    ap.add_argument("--seed", type=int, help=f"the rung's sampling seed (default {SEED}; the reference keeps {SEED}) --"
                    " a model that samples its speech is read over several seeds, not one")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--bin", help="llama.cpp bin folder")
    ap.add_argument("--keep", help="keep the wavs in this folder")
    ap.add_argument("--out", help="write the JSON report here")
    a = ap.parse_args(argv)
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    res = run(a)
    for k in ("speak_cer", "speak_dur_drift", "listen_cer", "long_cer", "long_repeat", "long_dur_drift"):
        if k in res:
            print(f"{k}: mean {res[k]['mean']}  worst {res[k]['worst']} ({res[k]['worst_lang']})")
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
