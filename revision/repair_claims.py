"""Idempotent editorial migration for the 12 Nov validity audit; changes no scores."""
from pathlib import Path
import re


def main():
    path = Path(__file__).resolve().parent / 'paper_wip/main.tex'
    text = path.read_text(encoding='utf-8')
    # Historical result IDs remain untouched; visible algorithm names disclose variants.
    for name, label in [('TENT', 'TENT-R'), ('EATA', 'EATA-inspired'),
                        ('SAR', 'SAR-inspired'), ('CoTTA', 'CoTTA-inspired')]:
        if name == 'SAR':
            # SAR also means the imaging modality: only change baseline lists/references.
            for old, new in [('EATA-inspired, SAR and', 'EATA-inspired, SAR-inspired and'),
                             ('EATA-inspired, SAR,', 'EATA-inspired, SAR-inspired,'),
                             ('and SAR \\citep{niu2023sar}', 'and SAR-inspired \\citep{niu2023sar}'),
                             ('over SAR,', 'over SAR-inspired,')]:
                text = text.replace(old, new)
        else:
            text = re.sub(r'\b' + name + r'\b(?![-\w])', label, text)
    text = text.replace('unlike reference TENT-R,', 'unlike reference TENT,')
    for old, new in [
        ('matched miss rates', 'common miss-rate targets'),
        ('a matched miss rate', 'a common miss-rate target'),
        ('the matched miss rate', 'the common miss-rate target'),
        ('every matched miss rate', 'every evaluated miss-rate target'),
        ('pre-registered test', 'internally planned test'),
        ('pre-registered final test', 'internally planned final test'),
        ('pre-registered MMFlood test', 'internally planned MMFlood test'),
        ('is level with it (0.677;', 'has an observed IoU of 0.677 versus 0.657; equivalence was not tested ('),
        ('and is level with supervised fine-tuning', 'and has a similar observed score to the supervised reference'),
        ('None of these methods checks pseudo-labels against physics, and none is evaluated at a controlled miss rate.',
         'Topography-guided adaptation has also been studied by \\citet{song2026tgtta}, whose framework uses soft teacher guidance and HAND/slope terrain-risk scores. Our distinction is the particular radar-and-terrain pseudo-label augmentation rule and the event-level calibration study, not the first use of terrain during test-time adaptation. A full implementation-level comparison remains necessary.'),
        ('A model-free reference confirms that MMFlood is hard rather than mishandled:',
         'A model-free reference also performs poorly, although this does not rule out preprocessing problems:'),
        ('so neither the unit conversion nor the resampling misaligns images and labels.',
         'which provides a limited preprocessing check, not proof that every image and label is aligned.'),
        ('Such water is invisible to any method that relies on single-date backscatter, including our physics check.',
         'These observations challenge our dark-water prior; they do not establish that all single-date SAR methods must fail.'),
        ('which suggests that the ceiling is set by the mismatch between MMFlood\'s images and labels and our single-date radar input rather than by model capacity.',
         'which leaves image/label mismatch, acquisition conditions, preprocessing and model limitations as competing explanations; it does not establish an accuracy ceiling.'),
        ('and that neither ImageNet pretraining nor 33 additional training floods lifted the MMFlood ceiling.',
         'but neither ImageNet pretraining nor 33 additional training floods substantially improved the observed MMFlood scores.'),
        ('the per-flood rule delivered the promised share of floods.',
         'the per-flood rule met the nominal event fraction empirically.'),
        ('and its miss rates are more even:', 'and its observed miss rates are more even:'),
        ('none of them beats the unadapted model', 'none has an established significant gain over the unadapted model'),
        ('none of the six baselines improves on the unadapted model', 'none of the six historical variants has an established significant gain over the unadapted model'),
        ('none of them gains anything', 'none has an established significant gain'),
        ('all of them realise the same mean miss rate', 'their mean miss rates are approximately equal'),
        ('adaptation is what makes a threshold calibrated on the source floods valid:',
         'adaptation changes whether the observed mean meets the source-calibrated target:'),
        ('because the shift between the two datasets is largely a shift in radiometry and land cover that batch-norm statistics absorb; the model\'s scores then mean roughly the same on both datasets.',
         'which is consistent with a role for radiometric or feature-distribution shifts; it does not establish their cause or prove that the scores have identical meanings across datasets.'),
        ('The safeguards that EATA-inspired, SAR-inspired and CoTTA-inspired add to entropy minimisation and self-training protect against collapse but do not add information about where the water is;',
         'The historical variants omit safeguards from their reference algorithms, so this comparison does not isolate the effects of those published safeguards;'),
        ('used Claude (Anthropic) to assist', 'used Claude (Anthropic) and Codex (OpenAI) to assist'),
    ]:
        text = text.replace(old, new)
    anchor = '\\paragraph{Where the calibration floods come from.}'
    caveat = ('\\paragraph{Scope of the theoretical statements.} The statements above require exchangeable loss curves under a fixed prediction/adaptation procedure. The Sen1Floods11 curves here are generated by different leave-one-event-out fitted models, while external events use the source-trained models. We report this cross-fitted transfer empirically; we have not established that the standard split-conformal proof covers this exact fitted-model protocol. The per-flood probability is marginal over calibration and a new event, not a conditional guarantee for every particular flood or a high-confidence guarantee for a realised calibration set.\n\n')
    if '\\paragraph{Scope of the theoretical statements.}' not in text:
        text = text.replace(anchor, caveat + anchor)
    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
