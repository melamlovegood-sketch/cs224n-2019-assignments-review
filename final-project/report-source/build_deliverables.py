"""Build the proposal, milestone, final report, and poster PDFs."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RESULTS_DIR = PROJECT_ROOT / "project" / "results"
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = json.loads((RESULTS_DIR / "results.json").read_text(encoding="utf-8"))
ERRORS = json.loads((RESULTS_DIR / "error_analysis.json").read_text(encoding="utf-8"))

NAVY = colors.HexColor("#12263A")
BLUE = colors.HexColor("#2563EB")
TEAL = colors.HexColor("#0F766E")
GOLD = colors.HexColor("#F59E0B")
LIGHT = colors.HexColor("#F3F6FA")
MID = colors.HexColor("#D6DEE8")
INK = colors.HexColor("#17202A")
MUTED = colors.HexColor("#52606D")


def _register_fonts() -> tuple[str, str]:
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("C:/Windows/Fonts/calibri.ttf"), Path("C:/Windows/Fonts/calibrib.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ProjectSans", str(regular)))
            pdfmetrics.registerFont(TTFont("ProjectSans-Bold", str(bold)))
            return "ProjectSans", "ProjectSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()


def styles():
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=sample["Title"], fontName=FONT_BOLD, fontSize=22, leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=14),
        "subtitle": ParagraphStyle("Subtitle", parent=sample["Normal"], fontName=FONT, fontSize=11, leading=15, textColor=MUTED, spaceAfter=16),
        "h1": ParagraphStyle("H1", parent=sample["Heading1"], fontName=FONT_BOLD, fontSize=16, leading=20, textColor=NAVY, spaceBefore=5, spaceAfter=9),
        "h2": ParagraphStyle("H2", parent=sample["Heading2"], fontName=FONT_BOLD, fontSize=12, leading=15, textColor=TEAL, spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=sample["BodyText"], fontName=FONT, fontSize=9.4, leading=13.2, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=sample["BodyText"], fontName=FONT, fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=4),
        "caption": ParagraphStyle("Caption", parent=sample["BodyText"], fontName=FONT, fontSize=7.5, leading=9.5, textColor=MUTED, alignment=TA_CENTER, spaceAfter=7),
        "callout": ParagraphStyle("Callout", parent=sample["BodyText"], fontName=FONT_BOLD, fontSize=10, leading=14, textColor=NAVY, backColor=LIGHT, borderColor=MID, borderWidth=0.7, borderPadding=9, spaceBefore=6, spaceAfter=10),
        "reference": ParagraphStyle("Reference", parent=sample["BodyText"], fontName=FONT, fontSize=8.2, leading=11.3, leftIndent=15, firstLineIndent=-15, textColor=INK, spaceAfter=6),
    }


S = styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"&#8226;&nbsp; {text}", ParagraphStyle("Bullet", parent=S["body"], leftIndent=14, firstLineIndent=-9, spaceAfter=5))


def document_header(story: list, title: str, subtitle: str) -> None:
    story.extend([P(title, "title"), P(subtitle, "subtitle"), Table([[""]], colWidths=[7.1 * inch], rowHeights=[0.05 * inch], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE)])), Spacer(1, 0.12 * inch)])


def page_decor(canvas_obj, doc) -> None:
    canvas_obj.saveState()
    canvas_obj.setStrokeColor(MID)
    canvas_obj.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas_obj.setFont(FONT, 7.5)
    canvas_obj.setFillColor(MUTED)
    canvas_obj.drawString(0.67 * inch, 0.34 * inch, "CS224N Default Final Project | Independent course study")
    canvas_obj.drawRightString(7.83 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas_obj.restoreState()


def metric_table() -> Table:
    baseline = RESULTS["baseline"]
    tuned = RESULTS["fine_tuned"]
    data = [
        ["System", "EM", "F1", "Ans. F1", "No-answer acc."],
        ["SQuAD 1.1 checkpoint", f"{baseline['default_metrics']['exact_match']:.2f}", f"{baseline['default_metrics']['f1']:.2f}", f"{baseline['default_metrics']['answerable_f1']:.2f}", f"{baseline['default_metrics']['unanswerable_accuracy']:.2f}"],
        ["Fine-tuned on SQuAD 2.0", f"{tuned['default_metrics']['exact_match']:.2f}", f"{tuned['default_metrics']['f1']:.2f}", f"{tuned['default_metrics']['answerable_f1']:.2f}", f"{tuned['default_metrics']['unanswerable_accuracy']:.2f}"],
        ["Fine-tuned + calibration", f"{tuned['calibrated_metrics']['exact_match']:.2f}", f"{tuned['calibrated_metrics']['f1']:.2f}", f"{tuned['calibrated_metrics']['answerable_f1']:.2f}", f"{tuned['calibrated_metrics']['unanswerable_accuracy']:.2f}"],
    ]
    table = Table(data, colWidths=[2.15 * inch, 0.72 * inch, 0.72 * inch, 0.82 * inch, 1.15 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#E6FFFA")),
        ("GRID", (0, 0), (-1, -1), 0.5, MID),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build_proposal() -> Path:
    path = OUTPUT_DIR / "project_proposal.pdf"
    story: list = []
    document_header(story, "Calibrated DistilBERT for Reliable Extractive QA", "Project proposal | Default project, PCE division | Independent course study")
    story += [P("Project overview", "h1"), P("This project studies a practical weakness of extractive question-answering systems: a model can locate plausible spans even when the passage does not support an answer. The central question is whether resource-efficient SQuAD 2.0 fine-tuning, followed by held-out calibration of the no-answer threshold, can improve both overall answer quality and abstention behavior on a laptop GPU."), P("The system starts from a compact DistilBERT checkpoint trained on SQuAD 1.1. That checkpoint is a deliberately informative baseline: it can extract answer spans, but it has not been trained on adversarial unanswerable questions. The proposed system keeps the architecture fixed, fine-tunes it on SQuAD 2.0, and calibrates the decision boundary on examples that are disjoint from final evaluation."), P("Planned contributions", "h2"), bullet("A reproducible preprocessing and training pipeline with sliding context windows and exact character-to-token alignment."), bullet("A controlled comparison separating the contribution of SQuAD 2.0 fine-tuning from post-hoc threshold calibration."), bullet("Answerable-only F1, unanswerable accuracy, and qualitative error analysis in addition to aggregate EM/F1."), P("Expected outcome", "h2"), P("We expect SQuAD 2.0 fine-tuning to produce the largest gain by teaching the model to use the [CLS] position as a no-answer alternative. Calibration should yield a smaller but measurable gain by correcting the operating point for the local evaluation distribution. The project will treat a negative calibration result as informative rather than selecting a threshold on the final test examples."), PageBreak()]

    document_header(story, "Research paper summary", "Rajpurkar, Jia, and Liang (2018): Know What You Don't Know")
    story += [P("Problem and motivation", "h1"), P("SQuAD 1.1 contains only answerable questions. A system can therefore improve by always finding the most plausible span, even when its evidence is weak. Rajpurkar et al. expose this limitation by adding more than 50,000 unanswerable questions written adversarially to resemble answerable ones. Success on SQuAD 2.0 requires both reading comprehension and selective prediction: the model must answer when evidence exists and abstain otherwise."), P("Dataset construction", "h2"), P("Crowdworkers were shown a paragraph and an existing answerable question, then asked to write a related question that the paragraph could not answer. This creates hard negatives with high lexical overlap and plausible distractors. The resulting benchmark combines these questions with the original SQuAD examples and evaluates both exact string match and token-level F1."), P("Main findings", "h2"), P("The paper shows a sharp gap between performance on SQuAD 1.1 and SQuAD 2.0. A strong neural reader that attained roughly 86 F1 on the original benchmark fell to about 66 F1 after unanswerable questions were introduced. The result demonstrates that span extraction alone is insufficient: robust systems need an explicit mechanism for comparing the best answer span with a no-answer option."), P("Connection to this project", "h2"), P("The proposed score difference follows the paper's central formulation. For each example, the system compares the [CLS] no-answer score with the highest valid span score. The threshold is not assumed to be zero; it is learned on a held-out calibration set, then frozen for final evaluation."), PageBreak()]

    document_header(story, "Methodological context", "Pretrained contextual representations and compact deployment")
    story += [P("BERT and extractive QA", "h1"), P("BERT represents the question and passage jointly with bidirectional self-attention. A small task-specific head predicts start and end logits for each input token. For unanswerable questions, both targets are assigned to [CLS], so the same logits support span extraction and abstention without a separate generative decoder."), P("Why DistilBERT", "h2"), P("DistilBERT compresses BERT through knowledge distillation. It retains much of BERT's language-understanding capability with fewer layers and lower inference cost. This makes it appropriate for a reproducible course project on an 8 GB laptop GPU while still testing the transfer-learning ideas discussed in the final-project handout."), P("Baseline and ablations", "h2"), bullet("Baseline A: SQuAD 1.1 checkpoint with the default threshold of zero."), bullet("Ablation B: the same checkpoint with threshold calibration only."), bullet("Model C: SQuAD 2.0 fine-tuning with the default threshold."), bullet("Final system D: SQuAD 2.0 fine-tuning plus held-out threshold calibration."), P("Evaluation design", "h2"), P("The SQuAD 2.0 validation set will be shuffled with a fixed seed and divided into a calibration subset and a non-overlapping final evaluation subset. The project will report EM, token F1, answerable-only EM/F1, and unanswerable accuracy. Qualitative examples will include both improvements and regressions."), PageBreak()]

    document_header(story, "Execution plan and risks", "Concrete milestones, compute budget, and success criteria")
    story += [P("Plan", "h1"), bullet("Restore the Stanford starter repository and preserve its BiDAF implementation for provenance."), bullet("Implement unit-tested normalization, span scoring, no-answer decisions, and threshold selection."), bullet("Run a tiny end-to-end smoke test before the full experiment."), bullet("Fine-tune for two epochs on 12,000 seeded training examples; use 1,500 examples for calibration and 1,500 for final evaluation."), bullet("Save the exact configuration, software versions, model checkpoint, predictions, figures, and error-analysis cases."), P("Success criteria", "h2"), P("The primary criterion is a clear improvement in held-out F1 over the unadapted SQuAD 1.1 checkpoint. Secondary criteria are increased unanswerable accuracy without collapsing answerable F1, a reproducible run on the available GPU, and a transparent analysis of failure modes."), P("Risks and mitigations", "h2"), P("A subset experiment may not match leaderboard performance; the report will avoid making that claim. Calibration may overfit if tuned on evaluation examples, so the split is fixed before inference and the final evaluation records are never used during threshold selection. Long contexts can truncate answers; a 64-token stride creates overlapping windows. Mixed precision can destabilize optimization; bfloat16 is used when supported, with gradient clipping and deterministic seeds."), P("Planned deliverables", "h2"), P("The final package will include source code, requirements, tests, model and tokenizer files, raw JSON metrics, predictions, training and metric plots, a 6-8 page report, and a research poster."), P("Key references", "h2"), P("Rajpurkar et al. (2018), Devlin et al. (2019), Sanh et al. (2019), and Seo et al. (2017). Full bibliographic entries are provided with the project source." )]
    SimpleDocTemplate(str(path), pagesize=LETTER, rightMargin=0.7 * inch, leftMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.7 * inch, title="CS224N Project Proposal").build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return path


def build_milestone() -> Path:
    path = OUTPUT_DIR / "project_milestone.pdf"
    story: list = []
    document_header(story, "Calibrated DistilBERT for Reliable Extractive QA", "Project milestone | Default project, PCE division | Independent course study")
    story += [P("Abstract", "h1"), P("We investigate reliable abstention for extractive question answering on SQuAD 2.0. A compact DistilBERT checkpoint trained only on SQuAD 1.1 provides strong span extraction but no useful default behavior on unanswerable questions. We implemented a complete SQuAD 2.0 pipeline, fine-tuned the model on 12,000 examples, and calibrated the no-answer threshold on 1,500 held-out examples. On a separate 1,500-example evaluation set, fine-tuning improved F1 from 41.79 to 63.92; calibration further increased it to 64.67. The final system obtains 62.53 EM, 64.67 F1, and 70.94% unanswerable accuracy."), P("Approach", "h1"), P("The model jointly encodes each question and context using a six-layer DistilBERT transformer. A linear QA head assigns start and end logits to every token. For each context window we enumerate valid spans from the top 20 start and end positions, retain the highest-scoring span up to 30 tokens, and compare it with the [CLS] no-answer score."), P("Long passages use 256-token windows with a 64-token stride. During training, answer spans outside a window and genuinely unanswerable examples both target [CLS]. During inference, the score difference d = s(null) - s(best span) controls abstention: the model emits an empty answer when d exceeds threshold tau."), P("Implementation status", "h2"), bullet("Data loading, token alignment, overflow windows, training, mixed precision, checkpointing, and plotting are complete."), bullet("Metric normalization and F1/EM logic are covered by unit tests; a 32-example smoke run passed end to end."), bullet("The full two-epoch GPU experiment and held-out evaluation are complete."), PageBreak()]

    document_header(story, "Experiments and preliminary results", "Seeded SQuAD 2.0 subset; final evaluation kept separate from calibration")
    story += [P("Experimental setup", "h1"), P("We sampled 12,000 training examples with seed 224, producing 13,322 overlapping features. A shuffled 3,000-example validation subset was split evenly: 1,500 examples for threshold selection and 1,500 for final evaluation. The latter contains 743 answerable and 757 unanswerable questions. We trained for two epochs using AdamW, learning rate 3e-5, weight decay 0.01, effective batch size 24, 10% warmup, gradient clipping at 1.0, and bfloat16 on an RTX 5060 Laptop GPU."), metric_table(), Spacer(1, 0.12 * inch), P("Preliminary interpretation", "h2"), P("The SQuAD 1.1 checkpoint answers every unanswerable evaluation question at the default threshold, so its unanswerable accuracy is 0%. Fine-tuning creates a meaningful no-answer signal: without calibration, unanswerable accuracy rises to 60.37% and overall F1 rises by 22.12 points. Threshold calibration moves tau to -1.828, raising unanswerable accuracy to 70.94% and overall F1 by a further 0.76 points."), Image(str(RESULTS_DIR / "training_loss.png"), width=6.4 * inch, height=3.65 * inch), P("Figure 1. Running loss across 1,111 optimizer steps; the apparent epoch reset reflects a per-epoch running mean.", "caption"), PageBreak()]

    document_header(story, "Progress, analysis, and remaining work", "What has been learned and what the final report will emphasize")
    story += [P("Qualitative progress", "h1"), P("Fine-tuning corrects both answer extraction and abstention. For the question 'What type of behavior in primes is it possible to determine?', the baseline abstains while the adapted model returns 'statistical', matching a reference. For the adversarial question 'What was the Yuan's Persian enemy?', the context mentions an ally in Persia but no enemy; the baseline chooses 'the Ilkhanate', whereas the adapted system correctly abstains."), P("Regressions remain. The adapted model sometimes selects a semantically complete but annotation-mismatched boundary, such as 'About 61.1%' instead of '61.1%'. It also becomes overconfident in abstention on some answerable examples. These cases motivate reporting token F1 beside exact match and separating answerable from unanswerable results."), P("Progress against plan", "h2"), bullet("Completed: repository recovery, modern pipeline, tests, smoke run, full training, calibration, independent evaluation, model checkpoint, plots, and error records."), bullet("Completed: quantitative comparison of three operating points and breakdown by question answerability."), bullet("In progress at milestone: converting the evidence into the final report and poster."), P("Final-report plan", "h2"), P("The final report will clearly distinguish architecture transfer, task adaptation, and calibration. It will include the aggregate table, training curve, selected improvements and regressions, limitations of subset evaluation, compute cost, and a reproducibility checklist. No leaderboard claim will be made because the experiment uses a seeded subset of the public validation data rather than the course's hidden test split."), P("Current conclusion", "h2"), P("The strongest result is not merely the 22.88-point F1 increase from task adaptation; it is the change in error balance. Calibration trades some answerable recall for much better abstention, and the disaggregated metrics make that trade-off visible. This is the central lesson to carry into the final analysis.")]
    SimpleDocTemplate(str(path), pagesize=LETTER, rightMargin=0.7 * inch, leftMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.7 * inch, title="CS224N Project Milestone").build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return path


def build_final_report() -> Path:
    path = OUTPUT_DIR / "final_report.pdf"
    story: list = []
    document_header(story, "Calibrated DistilBERT for Reliable Extractive Question Answering", "Final report | Default project, PCE division | Independent CS224N course study")
    story += [P("Abstract", "h1"), P("Extractive question-answering models can return plausible spans even when a passage does not support an answer. We evaluate a compact DistilBERT reader originally trained on SQuAD 1.1, adapt it to adversarial unanswerable questions from SQuAD 2.0, and calibrate its no-answer decision on data disjoint from final evaluation. The experiment uses 12,000 training examples, 1,500 calibration examples, and 1,500 held-out evaluation examples. SQuAD 2.0 fine-tuning raises Exact Match from 38.00 to 61.27 and token F1 from 41.79 to 63.92. Calibration further raises the scores to 62.53 EM and 64.67 F1 while increasing unanswerable accuracy from 60.37% to 70.94%. Error analysis shows that task adaptation corrects unsupported guesses and recovers answer spans, but threshold selection can reduce answerable recall and boundary errors persist. The results support calibrated abstention as a cheap, transparent complement to task-specific fine-tuning."), P("1. Introduction", "h1"), P("Question answering is useful only when a system can distinguish evidence from a convincing distractor. SQuAD 1.1 rewards selecting a span for every question, which lets a model succeed without learning when to abstain. SQuAD 2.0 directly targets this failure by adding adversarial questions whose words and topics overlap with a paragraph although no answer is stated."), P("This project asks two questions: How much does SQuAD 2.0 adaptation improve a compact model trained on answerable questions, and does a no-answer threshold chosen on held-out calibration data improve generalization? We answer them with one architecture and controlled interventions, avoiding a comparison in which parameter count changes at the same time as the training objective."), P("Our contribution is a reproducible laptop-scale experiment that separates three effects: the inherited SQuAD 1.1 span reader, task adaptation to SQuAD 2.0, and post-hoc decision calibration. We report not only aggregate EM and F1 but also answerable-only quality and unanswerable accuracy, revealing an important precision-recall trade-off hidden by the headline metric."), PageBreak()]

    document_header(story, "2. Related work", "Extractive QA, adversarial unanswerability, and efficient contextual encoders")
    story += [P("SQuAD and unanswerable questions", "h1"), P("The original Stanford Question Answering Dataset framed reading comprehension as extractive span selection over Wikipedia passages. SQuAD 2.0 adds more than 50,000 adversarial unanswerable questions. Rajpurkar et al. showed that a reader with roughly 86 F1 on SQuAD 1.1 fell to about 66 F1 on SQuAD 2.0, demonstrating that locating plausible spans is not equivalent to recognizing textual support."), P("Neural readers", "h2"), P("BiDAF introduced a multi-stage architecture with contextual word representations and bidirectional attention between question and context. The supplied CS224N baseline follows this design. It is historically important and remains in the repository for provenance, but its word-level GloVe pipeline and recurrent encoders carry a larger setup burden in a modern Windows environment."), P("Pretrained contextual encoders", "h2"), P("BERT pretrains a bidirectional Transformer and adapts it to extractive QA with a small start/end prediction head. Joint question-context attention is available at every layer, and [CLS] supplies a natural null location for SQuAD 2.0. DistilBERT compresses the teacher using language-modeling, distillation, and cosine losses. The reported model is smaller and faster than BERT while retaining much of its language-understanding quality, making it a practical PCE choice for constrained hardware."), P("Selective prediction and calibration", "h2"), P("A SQuAD 2.0 reader compares evidence for the best span with evidence for no answer. That score difference is not itself a calibrated probability, and a zero threshold need not maximize an application metric. Selecting the threshold on separate calibration data is a simple form of operating-point calibration. The procedure changes no model parameters and is therefore easy to audit, reproduce, and deploy."), P("Research gap addressed here", "h2"), P("Many comparisons conflate architecture size, pretraining, task data, and decoding. Our experiment holds architecture and starting checkpoint constant, then measures the effects of task fine-tuning and threshold selection separately. The emphasis is less on state-of-the-art rank than on causal clarity under a realistic compute budget."), PageBreak()]

    document_header(story, "3. Approach", "Span extraction, SQuAD 2.0 supervision, and held-out threshold selection")
    story += [P("3.1 Model", "h1"), P("The encoder is distilbert-base-cased, a six-layer bidirectional Transformer with approximately 66.4 million parameters. Input is [CLS] question [SEP] context [SEP]. A learned linear layer produces start logits l_s(i) and end logits l_e(i) for each token i. The score for span (i,j) is l_s(i)+l_e(j), subject to j >= i and a maximum length of 30 tokens."), P("3.2 Long-context processing", "h2"), P("Inputs are capped at 256 wordpiece tokens. Contexts that exceed this limit are split into overlapping windows with a stride of 64 tokens. Character offsets retained by the fast tokenizer map predicted token positions back to exact substrings. For every example, candidate spans are pooled across all windows and the globally highest-scoring valid span is retained."), P("3.3 No-answer learning", "h2"), P("For an unanswerable training example, both gold positions are [CLS]. The same target is used for a window that does not contain the gold answer, preventing a partial window from being labeled with a false span. Let s_null denote the sum of start and end logits at [CLS] and s_span the best valid span score. We define d = s_null - s_span and predict no answer if d > tau."), P("3.4 Threshold calibration", "h2"), P("Threshold tau is selected to maximize token F1 on 1,500 calibration examples. The search evaluates every distinct observed score difference efficiently by sorting examples and updating the total metric when an example switches from null to non-null. The chosen threshold is then frozen. A separate 1,500-example evaluation split is used exactly once for final reporting."), P("3.5 Training objective", "h2"), P("We minimize the mean of cross-entropy losses for start and end positions, beginning with a checkpoint already fine-tuned on SQuAD 1.1. AdamW uses learning rate 3e-5, weight decay 0.01, 10% linear warmup, and gradient clipping at 1.0. Two epochs are trained with an effective batch size of 24. Bfloat16 autocasting reduces memory use without loss scaling on the available GPU."), P("3.6 Reproducibility", "h2"), P("Seed 224 controls dataset shuffling, data-loader order, NumPy, Python, and PyTorch. The pipeline records exact package versions, device, sample counts, feature counts, wall-clock timing, hyperparameters, training history, predictions, and selected error examples in machine-readable JSON."), PageBreak()]

    document_header(story, "4. Experimental setup", "A resource-bounded, disjoint-split evaluation")
    story += [P("4.1 Data", "h1"), P("We use the public Hugging Face mirror of SQuAD 2.0, which contains 130,319 training examples and 11,873 validation examples. From training we select 12,000 examples after seeded shuffling; token overflow produces 13,322 model features. From validation we select 3,000 examples and divide them evenly into calibration and final evaluation sets. The evaluation set contains 743 answerable and 757 unanswerable questions and produces 1,683 model features."), P("4.2 Systems", "h2"), bullet("Unadapted baseline: the SQuAD 1.1 DistilBERT checkpoint, tau = 0."), bullet("Calibrated baseline: the same frozen checkpoint, with tau selected on calibration data."), bullet("Adapted model: two epochs of SQuAD 2.0 fine-tuning, tau = 0."), bullet("Final system: the adapted model with its separately calibrated threshold."), P("4.3 Metrics", "h2"), P("Exact Match normalizes case, punctuation, articles, and whitespace before requiring equality with any reference. Token F1 computes overlap between normalized predicted and gold tokens and takes the maximum over references. Empty gold answers represent unanswerable questions. We additionally report answerable-only EM/F1 and the fraction of unanswerable questions correctly assigned an empty prediction."), P("4.4 Compute", "h2"), P("Experiments ran with Python 3.13.7, PyTorch 2.10.0, Transformers 5.17.0, and Datasets 5.0.1 on an NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB memory. Baseline evaluation took 11.5 seconds, training took 274.0 seconds, and final evaluation took 15.0 seconds. These times exclude the one-time model and dataset downloads."), P("4.5 Scope", "h2"), P("The experiment is designed for reproducible course completion, not direct comparison with the original private class leaderboard. It uses a subset of the public validation set and a 256-token limit. Results should therefore be interpreted as an internal controlled comparison. All systems see the identical final examples, and no final label is used for training or calibration."), Image(str(RESULTS_DIR / "training_loss.png"), width=3.7 * inch, height=2.11 * inch), P("Figure 1. Training loss over two epochs.", "caption"), PageBreak()]

    document_header(story, "5. Results", "Task adaptation supplies the main gain; calibration improves the operating point")
    story += [metric_table(), Spacer(1, 0.10 * inch), Image(str(RESULTS_DIR / "metric_comparison.png"), width=5.35 * inch, height=3.01 * inch), P("Figure 2. Held-out aggregate EM and F1. The chart separates the default and calibrated operating points of the adapted model.", "caption"), P("5.1 Aggregate performance", "h2"), P("SQuAD 2.0 fine-tuning increases EM by 23.27 points and F1 by 22.12 points relative to the unadapted default-threshold baseline. Calibration adds 1.27 EM and 0.76 F1 on the final set. The calibrated final system reaches 62.53 EM and 64.67 F1."), P("5.2 Abstention behavior", "h2"), P("The unadapted checkpoint has 0% unanswerable accuracy at tau = 0 because it was trained to answer every SQuAD 1.1 question. Fine-tuning raises that accuracy to 60.37%. Calibration sets tau = -1.828 and raises it again to 70.94%, but answerable F1 falls from 67.53 to 58.29. Thus the aggregate gain comes from a deliberate trade: more correct abstentions outweigh lost answerable recall on this nearly balanced set."), P("5.3 Calibration ablation", "h2"), P("Calibration alone raises the frozen baseline from 41.79 to 57.51 F1, but its answerable F1 drops from 84.37 to 46.11. This shows that a threshold can mask a model's lack of task adaptation by exploiting class balance. The adapted model is preferable because it maintains substantially stronger answerable performance while attaining similar no-answer accuracy."), PageBreak()]

    document_header(story, "6. Qualitative analysis", "Corrected unsupported guesses, recovered spans, and remaining boundary errors")
    story += [P("6.1 Successful answer recovery", "h1"), P("The baseline sometimes over-abstains after calibration because its score distribution was learned without unanswerable training examples. On the Prime number article, it returns an empty answer for 'What type of behavior in primes is it possible to determine?' The adapted model returns 'statistical', matching one of the references. Similar recoveries include 'the heart' for the organ whose workload is eased by reduced pulmonary resistance and 'Germany' for the Ottoman Empire's World War I ally."), P("6.2 Correct abstention", "h2"), P("Adversarial questions often invert a relation or alter a premise. The Yuan question asks for a Persian enemy, while the passage mentions the Ilkhanate as an ally. The baseline selects the nearby entity; the adapted model abstains. Another question asks for a least-important theoretical-computer-science problem although the passage calls P versus NP one of the most important. Fine-tuning correctly maps this contradiction to no answer."), P("6.3 Boundary regressions", "h2"), P("The adapted reader sometimes predicts a semantically valid superset: 'About 61.1%' instead of '61.1%', '13 years old' instead of '13', and 'rounded arches' instead of 'rounded'. Exact Match scores these as wrong even though token F1 gives partial credit. These cases suggest that start/end calibration and answer-length priors could complement no-answer calibration."), P("6.4 Over-abstention regressions", "h2"), P("The model also abstains on some answerable questions. For a Rhine passage, the baseline extracts '120 m (390 ft)' while the adapted calibrated model emits no answer; the gold is '120 m'. A long European Union law example similarly triggers abstention even though the relevant court case appears late in the passage. The likely causes are a conservative threshold, long-context competition, and limited subset fine-tuning."), P("6.5 Error taxonomy", "h2"), bullet("Unsupported entity selection: improved strongly after SQuAD 2.0 training."), bullet("Reversed or impossible premise: often corrected through abstention."), bullet("Span-boundary mismatch: persists after adaptation."), bullet("Long-context evidence and over-abstention: remains a major failure mode."), P("The saved error-analysis file contains twelve improvements and eight regressions with contexts, predictions, references, and per-example scores, enabling the claims above to be audited."), PageBreak()]

    document_header(story, "7. Discussion and conclusion", "What the controlled comparison shows—and what it does not")
    story += [P("7.1 Main finding", "h1"), P("Task-specific exposure to adversarial unanswerable questions is the dominant improvement. The inherited SQuAD 1.1 reader knows how to find spans but not when evidence is absent. Fine-tuning creates a useful null score and improves both aggregate quality and abstention. A separately chosen threshold then provides a smaller, inexpensive improvement in final EM/F1."), P("7.2 Why disaggregated metrics matter", "h2"), P("Aggregate F1 alone would make calibration look uniformly beneficial. The answerability breakdown reveals that its 0.76-point overall gain is purchased with a 9.23-point reduction in answerable F1 and a 10.57-point increase in unanswerable accuracy. Applications should therefore choose tau using domain costs: a medical assistant may prefer conservative abstention, while a search interface may prefer higher answer recall."), P("7.3 Limitations", "h2"), bullet("Only 12,000 of 130,319 training examples are used, limiting convergence and rare-pattern coverage."), bullet("The final set has 1,500 public validation examples rather than the private course leaderboard test set."), bullet("A single random seed does not quantify training variance."), bullet("Threshold calibration optimizes F1 and does not measure probability calibration such as expected calibration error."), bullet("Maximum sequence length 256 can fragment long evidence despite overlapping windows."), P("7.4 Future work", "h2"), P("A stronger follow-up would train on the full dataset with multiple seeds, compare global and answer-type-specific thresholds, add a calibrated answerability head, and examine length penalties for boundary precision. Long-context transformers or retrieval could reduce failures on evidence appearing late in a passage. Temperature scaling or conformal prediction could provide more principled confidence guarantees."), P("7.5 Conclusion", "h2"), P("On a fixed held-out SQuAD 2.0 evaluation subset, two epochs of compact-model adaptation raise F1 from 41.79 to 63.92, and held-out no-answer calibration raises it to 64.67. The final system is not a state-of-the-art leaderboard entry, but it is a complete, reproducible demonstration of a central reliability lesson: answer extraction and the decision to answer are distinct capabilities, and both require explicit evaluation."), P("Reproducibility checklist", "h2"), P("Code, dependency list, unit tests, exact config, software versions, model checkpoint, tokenizer, raw metrics, training history, predictions, figures, and qualitative cases are included. The code submission excludes model weights and data, following course instructions."), PageBreak()]

    document_header(story, "References", "Bibliography and implementation sources")
    refs = [
        "[1] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP, 2016. https://arxiv.org/abs/1606.05250",
        "[2] P. Rajpurkar, R. Jia, and P. Liang. Know What You Don't Know: Unanswerable Questions for SQuAD. ACL, 2018. https://arxiv.org/abs/1806.03822",
        "[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019. https://arxiv.org/abs/1810.04805",
        "[4] V. Sanh, L. Debut, J. Chaumond, and T. Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. NeurIPS EMC2 Workshop, 2019. https://arxiv.org/abs/1910.01108",
        "[5] M. Seo, A. Kembhavi, A. Farhadi, and H. Hajishirzi. Bidirectional Attention Flow for Machine Comprehension. ICLR, 2017. https://arxiv.org/abs/1611.01603",
        "[6] Hugging Face. Transformers: Question Answering task guide. https://huggingface.co/docs/transformers/main/tasks/question_answering",
        "[7] Hugging Face / Rajpurkar et al. SQuAD 2.0 dataset card. https://huggingface.co/datasets/rajpurkar/squad_v2",
        "[8] C. Chute. Stanford CS224N SQuAD 2.0 starter code. https://github.com/chrischute/squad",
    ]
    for ref in refs:
        story.append(P(ref, "reference"))
    story += [P("Artifact note", "h2"), P("This independent course-completion report was produced from the exact machine-readable results stored with the project. No private leaderboard score, team identity, or institutional email address is claimed."), P("Model and data licenses", "h2"), P("The project uses the public DistilBERT checkpoint and the SQuAD 2.0 dataset through their respective Hugging Face repositories. Users should consult the associated model card, dataset card, and upstream licenses before redistribution.")]
    SimpleDocTemplate(str(path), pagesize=LETTER, rightMargin=0.7 * inch, leftMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.7 * inch, title="Calibrated DistilBERT for Reliable Extractive Question Answering").build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return path


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, width: float, font: str, size: float, leading: float, color=INK) -> float:
    style = ParagraphStyle("PosterText", fontName=font, fontSize=size, leading=leading, textColor=color)
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, 20 * inch)
    paragraph.drawOn(c, x, y - height)
    return y - height


def poster_box(c: canvas.Canvas, x: float, y: float, width: float, height: float, title: str) -> tuple[float, float]:
    c.setFillColor(colors.white)
    c.setStrokeColor(MID)
    c.setLineWidth(2)
    c.roundRect(x, y - height, width, height, 12, fill=1, stroke=1)
    c.setFillColor(NAVY)
    c.roundRect(x, y - 0.65 * inch, width, 0.65 * inch, 12, fill=1, stroke=0)
    c.rect(x, y - 0.65 * inch, width, 0.25 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(FONT_BOLD, 22)
    c.drawString(x + 0.25 * inch, y - 0.43 * inch, title)
    return x + 0.28 * inch, y - 0.9 * inch


def build_poster() -> Path:
    path = OUTPUT_DIR / "project_poster.pdf"
    page = (30 * inch, 20 * inch)
    c = canvas.Canvas(str(path), pagesize=page)
    c.setFillColor(LIGHT)
    c.rect(0, 0, page[0], page[1], fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, 17.25 * inch, page[0], 2.75 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(FONT_BOLD, 42)
    c.drawString(0.8 * inch, 18.75 * inch, "Calibrated DistilBERT for Reliable Extractive QA")
    c.setFont(FONT, 22)
    c.drawString(0.82 * inch, 18.15 * inch, "SQuAD 2.0 task adaptation + held-out no-answer calibration")
    c.setFont(FONT, 14)
    c.drawRightString(29.2 * inch, 18.18 * inch, "CS224N default final project | Independent course study")

    margin = 0.65 * inch
    gap = 0.35 * inch
    col = (page[0] - 2 * margin - 2 * gap) / 3
    top = 16.85 * inch

    x1 = margin
    bx, by = poster_box(c, x1, top, col, 7.45 * inch, "Problem & research question")
    by = draw_wrapped(c, "Extractive QA models can return a convincing span even when a passage does not contain an answer. SQuAD 2.0 makes this failure explicit with adversarial unanswerable questions.", bx, by, col - 0.56 * inch, FONT, 16, 21)
    by -= 0.25 * inch
    by = draw_wrapped(c, "<b>Question:</b> How much do SQuAD 2.0 fine-tuning and a separately calibrated no-answer threshold improve a compact model trained only on SQuAD 1.1?", bx, by, col - 0.56 * inch, FONT, 16, 21, TEAL)
    by -= 0.28 * inch
    by = draw_wrapped(c, "<b>Controlled design</b><br/>1. Same 66.4M-parameter DistilBERT architecture.<br/>2. Measure task adaptation separately from threshold calibration.<br/>3. Never use final evaluation labels to select the threshold.", bx, by, col - 0.56 * inch, FONT, 15, 20)

    bx, by = poster_box(c, x1, 8.95 * inch, col, 8.2 * inch, "Data & experimental setup")
    by = draw_wrapped(c, "<b>Train:</b> 12,000 examples / 13,322 windowed features<br/><b>Calibration:</b> 1,500 examples<br/><b>Final evaluation:</b> 1,500 examples (743 answerable, 757 unanswerable)<br/><b>Input:</b> 256 wordpieces, 64-token stride<br/><b>Training:</b> 2 epochs, AdamW, lr 3e-5, effective batch 24, bfloat16<br/><b>Hardware:</b> RTX 5060 Laptop GPU, 8 GB<br/><b>Training time:</b> 274 seconds", bx, by, col - 0.56 * inch, FONT, 15, 21)
    by -= 0.25 * inch
    by = draw_wrapped(c, "<b>Metrics:</b> normalized Exact Match, token F1, answerable-only F1, and unanswerable accuracy.", bx, by, col - 0.56 * inch, FONT, 15, 20, TEAL)

    x2 = margin + col + gap
    bx, by = poster_box(c, x2, top, col, 10.6 * inch, "Method")
    by = draw_wrapped(c, "Question and context are encoded jointly. A QA head predicts start and end logits for every token. Candidate spans are pooled across overlapping context windows.", bx, by, col - 0.56 * inch, FONT, 16, 21)
    by -= 0.25 * inch
    formula = "<b>span score</b> = l<sub>s</sub>(i) + l<sub>e</sub>(j)<br/><b>null score</b> = l<sub>s</sub>([CLS]) + l<sub>e</sub>([CLS])<br/><b>d</b> = null score - best span score<br/><b>abstain when</b> d &gt; tau"
    by = draw_wrapped(c, formula, bx, by, col - 0.56 * inch, FONT, 18, 27, NAVY)
    by -= 0.3 * inch
    by = draw_wrapped(c, "<b>Fine-tuning</b> assigns both gold positions to [CLS] for unanswerable examples and windows without the answer. <b>Calibration</b> selects tau on a disjoint split, freezes it, then evaluates once on held-out data.", bx, by, col - 0.56 * inch, FONT, 16, 22)
    chart = Image(str(RESULTS_DIR / "training_loss.png"), width=col - 0.7 * inch, height=(col - 0.7 * inch) * 0.57)
    chart.wrapOn(c, col, 5 * inch)
    chart.drawOn(c, x2 + 0.35 * inch, top - 9.9 * inch)

    bx, by = poster_box(c, x2, 5.85 * inch, col, 5.1 * inch, "Reproducibility")
    draw_wrapped(c, "Seeded split, unit tests, smoke run, exact configuration, model checkpoint, tokenizer, software versions, raw metrics, predictions, figures, and 20 auditable error cases are included. The code package excludes data and model weights.", bx, by, col - 0.56 * inch, FONT, 15, 21)

    x3 = margin + 2 * (col + gap)
    bx, by = poster_box(c, x3, top, col, 9.7 * inch, "Results")
    metrics_img = Image(str(RESULTS_DIR / "metric_comparison.png"), width=col - 0.55 * inch, height=(col - 0.55 * inch) * 0.5625)
    metrics_img.wrapOn(c, col, 6 * inch)
    metrics_img.drawOn(c, x3 + 0.27 * inch, top - 5.9 * inch)
    by = top - 6.45 * inch
    draw_wrapped(c, "<b>Fine-tuning gain:</b> +23.27 EM / +22.12 F1<br/><b>Calibration gain:</b> +1.27 EM / +0.76 F1<br/><b>Final:</b> 62.53 EM / 64.67 F1 / 70.94% no-answer accuracy", bx, by, col - 0.56 * inch, FONT_BOLD, 17, 24, TEAL)

    bx, by = poster_box(c, x3, 6.7 * inch, col, 5.95 * inch, "Error analysis & conclusion")
    by = draw_wrapped(c, "<b>Corrected:</b> unsupported entities and reversed premises. Example: a passage calls the Ilkhanate an ally, while the adversarial question asks for a Persian enemy; the final system abstains.", bx, by, col - 0.56 * inch, FONT, 15, 20)
    by -= 0.16 * inch
    by = draw_wrapped(c, "<b>Remaining:</b> span-boundary mismatches ('About 61.1%' vs. '61.1%'), long-context misses, and over-abstention.", bx, by, col - 0.56 * inch, FONT, 15, 20)
    by -= 0.2 * inch
    draw_wrapped(c, "<b>Takeaway:</b> Answer extraction and the decision to answer are distinct capabilities. SQuAD 2.0 adaptation supplies the main improvement; held-out calibration cheaply adjusts the reliability/recall trade-off.", bx, by, col - 0.56 * inch, FONT_BOLD, 16, 22, NAVY)

    c.setFont(FONT, 10)
    c.setFillColor(MUTED)
    c.drawString(0.7 * inch, 0.25 * inch, "Sources: Rajpurkar et al. 2018; Devlin et al. 2019; Sanh et al. 2019. Results are from a seeded public-validation subset, not the private leaderboard.")
    c.save()
    return path


def main() -> None:
    paths = [build_proposal(), build_milestone(), build_final_report(), build_poster()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
