// Deck for the SIMBig 2026 Task 1 video presentation.
// Style follows the reference slides: white ground, serif display titles,
// terracotta accent, beige stat cards, generous whitespace.
// Build:  node build_deck.js      Output: simbig2026_task1.pptx

const pptxgen = require("pptxgenjs");
const path = require("path");

const THEME = {
  name: "Puno Quechua ASR",
  headFontFace: "Cambria",          // serif display, safe list
  bodyFontFace: "Calibri",          // sans body, safe list
  colors: {
    dk1: "1A1A1A",   // ink
    lt1: "FFFFFF",   // ground
    dk2: "5F5F5F",   // muted text
    lt2: "F2EDE9",   // card beige
    accent1: "B5513A",  // terracotta
    accent2: "8C8C8C",
    accent3: "2F6B3C",
    accent4: "B5513A",
    accent5: "D8CFC8",
    accent6: "1A1A1A",
    hlink: "B5513A",
    folHlink: "8C8C8C",
  },
};

const RUST = THEME.colors.accent1;
const INK = THEME.colors.dk1;
const MUTED = THEME.colors.dk2;
const CARD = THEME.colors.lt2;
const GREEN = THEME.colors.accent3;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.3 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Raymond Negron; Rosa Y. G. Paccotacya-Yanque";
pres.title = "Puno Quechua ASR: ortografia, solapamiento y tres resultados negativos";

const M = 0.85;                          // side margin
const W = 13.33 - 2 * M;                 // usable width

// ---------------------------------------------------------------- layouts --
pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: M, y: 1.15, w: W, h: 0.3,
        fontSize: 12, bold: true, charSpacing: 2.2, color: RUST, fontFace: THEME.bodyFontFace, align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.6, w: W - 1.2, h: 1.7,
        fontSize: 40, color: INK, fontFace: THEME.headFontFace, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: M, y: 3.35, w: W - 1.2, h: 0.5,
        fontSize: 16, italic: true, color: MUTED, fontFace: THEME.headFontFace, align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "authors", type: "body", x: M, y: 4.0, w: W, h: 0.8,
        fontSize: 13, color: INK, fontFace: THEME.bodyFontFace, align: "left", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: "FFFFFF" },
  objects: [
    { line: { x: M, y: 0.58, w: 0.52, h: 0, line: { color: RUST, width: 2.5 } } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.74, w: W, h: 0.72,
        fontSize: 29, color: INK, fontFace: THEME.headFontFace, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 1.64, w: W, h: 0.9,
        fontSize: 15, color: MUTED, fontFace: THEME.bodyFontFace, valign: "top", align: "left", margin: 0, lineSpacingMultiple: 1.2 }, text: "" } },
  ],
  slideNumber: { x: 12.55, y: 6.92, fontSize: 10, color: "AAAAAA", fontFace: THEME.bodyFontFace },
});

pres.defineSlideMaster({
  title: "CIERRE",
  background: { color: "FFFFFF" },
  objects: [
    { line: { x: M, y: 0.72, w: 0.52, h: 0, line: { color: RUST, width: 2.5 } } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.88, w: W, h: 1.0,
        fontSize: 34, color: INK, fontFace: THEME.headFontFace, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.55, y: 6.92, fontSize: 10, color: "AAAAAA", fontFace: THEME.bodyFontFace },
});

// ----------------------------------------------------------------- helpers --
// A row of beige cards: [{ big, label }]
function cards(slide, items, y, h = 1.45, opts = {}) {
  const gap = 0.3;
  const w = (W - gap * (items.length - 1)) / items.length;
  items.forEach((it, i) => {
    const x = M + i * (w + gap);
    slide.addShape(pres.ShapeType.roundRect, {
      x, y, w, h, rectRadius: 0.06, fill: { color: opts.fill || CARD },
      line: { color: opts.fill || CARD }, objectName: `card-${y}-${i}`,
    });
    slide.addText(it.big, {
      x: x + 0.28, y: y + 0.14, w: w - 0.5, h: 0.55, isTextBox: true, margin: 0,
      fontSize: opts.bigSize || 30, color: it.color || RUST, fontFace: THEME.headFontFace, valign: "middle",
    });
    slide.addText(it.label, {
      x: x + 0.28, y: y + 0.72, w: w - 0.5, h: h - 0.85, isTextBox: true, margin: 0,
      fontSize: 12, color: MUTED, fontFace: THEME.bodyFontFace, valign: "top", lineSpacingMultiple: 1.15,
    });
  });
}

// Three takeaway columns at the foot of a slide
function takeaways(slide, items, y) {
  const gap = 0.4;
  const w = (W - gap * (items.length - 1)) / items.length;
  items.forEach((it, i) => {
    const x = M + i * (w + gap);
    slide.addText(it.head, {
      x, y, w, h: 0.3, isTextBox: true, margin: 0,
      fontSize: 13, bold: true, color: RUST, fontFace: THEME.bodyFontFace,
    });
    slide.addText(it.body, {
      x, y: y + 0.32, w, h: 1.0, isTextBox: true, margin: 0,
      fontSize: 12, color: MUTED, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.2,
    });
  });
}

// Booktabs table, drawn by hand.
//
// A PowerPoint table carries a default style that Keynote — and some PowerPoint
// themes — paint over the cell borders, so a grid comes back however the
// borders are set. Composing the table from text boxes and three rules renders
// identically everywhere: a thick rule above the header, a hairline under it,
// and a thick rule at the foot. No vertical lines, no inner grid.
// Identifiers get their own face, the way LaTeX sets them apart from prose:
//   code()  monospace, for names you could type — sharedtask, mask_time_prob
//   mvar()  serif italic, for the single-letter conditions of an experiment,
//           which is how LaTeX renders a maths variable
const MONO = "Courier New";
const code = (t, o = {}) => ({ text: t, options: { fontFace: MONO, ...o } });
const mvar = (t, o = {}) => ({ text: t, options: { fontFace: THEME.headFontFace, italic: true, ...o } });
const txt = (t, o = {}) => ({ text: t, options: o });

// Numbers carry the argument, so they are set bold and a touch larger than the
// prose around them. Splits a sentence into runs and emphasises the figures.
function emph(sentence, o = {}) {
  const size = o.fontSize || 14;
  return sentence
    .split(/(\d[\d.,]*(?:\s?%)?|\b2e-5\b|\b1e-5\b|\b5e-5\b)/g)
    .filter((part) => part !== "")
    .map((part) =>
      /^\d/.test(part)
        ? { text: part, options: { ...o, bold: true, fontSize: size + 1.5 } }
        : { text: part, options: { ...o } }
    );
}

let tblN = 0;
function tbl(slide, rows, opts = {}) {
  tblN += 1;
  const x0 = opts.x !== undefined ? opts.x : M;
  const y0 = opts.y;
  const colW = opts.colW;
  const width = colW.reduce((a, b) => a + b, 0);
  const rowH = opts.rowH || 0.42;
  const fs = opts.fontSize || 14;
  const PAD = 0.08;                 // the breathing room \addlinespace gives
  const HEAD = rowH + PAD * 2;      // the band between \toprule and \midrule
  const midY = y0 + HEAD;
  const bodyY = midY + PAD;

  const rule = (y, pt, color) =>
    slide.addShape(pres.ShapeType.line, {
      x: x0, y, w: width, h: 0,
      line: { color, width: pt },
      objectName: `rule-${tblN}-${y.toFixed(2)}`,
    });

  rows.forEach((row, r) => {
    // The header fills its whole band, so valign centres it between the rules.
    const y = r === 0 ? y0 : bodyY + (r - 1) * rowH;
    const h = r === 0 ? HEAD : rowH;
    let cx = x0;
    row.forEach((cell, c) => {
      const o = typeof cell === "string" ? {} : cell.options || {};
      const text = typeof cell === "string" ? cell : cell.text;
      slide.addText(text, {
        x: cx, y, w: colW[c], h,
        // right margin on every cell, left margin on all but the first,
        // so a right-aligned figure never touches the label beside it
        // margin is in POINTS, not inches: ~11 pt of gutter each side
        isTextBox: true, margin: [0, 11, 0, c === 0 ? 1 : 11],
        // figures carry the result, so they run larger than the labels
        fontSize: o.align === "right" && r > 0 ? fs + 4.5 : fs,
        fontFace: THEME.bodyFontFace,
        color: o.color || INK, bold: o.bold || r === 0,
        align: o.align || "left", valign: "middle",
        objectName: `cell-${tblN}-${r}-${c}`,
      });
      cx += colW[c];
    });
  });

  const bottom = bodyY + (rows.length - 1) * rowH + PAD;
  rule(y0, 1.5, INK);               // \toprule
  rule(midY, 0.85, "6E625B");       // \midrule
  rule(bottom, 1.5, INK);           // \bottomrule
  return bottom;
}

// ===================================================================== 1 ====
let s = pres.addSlide({ masterName: "PORTADA" });
s.addText("SIMBIG 2026  ·  TAREA 1  ·  EQUIPO INNOVA", { placeholder: "eyebrow" });
s.addText("ASR para el quechua de Puno: ortografía, solapamiento y tres resultados negativos",
  { placeholder: "title" });
s.addText("Puno Quechua ASR: Orthography, Sentence Overlap, and Three Negative Results",
  { placeholder: "subtitle" });
s.addText([
  { text: "Raymond Negron  ·  Rosa Y. G. Paccotacya-Yanque", options: { bold: true, breakLine: true } },
  { text: "Department of Computing, Universidad Católica San Pablo  ·  Arequipa, Perú", options: { color: MUTED } },
], { placeholder: "authors" });
cards(s, [
  { big: "13,15 → 9,14", label: "WER medio frente al baseline publicado,\nmedido con la misma regla" },
  { big: "8 puntos", label: "de WER explicados solo por la ortografía\nde la referencia curada" },
  { big: "1,1 puntos", label: "lo que vale haber oído una frase\nen boca de otros hablantes" },
], 5.0, 1.6, { bigSize: 26 });
s.addNotes("Buenos días. Presento el sistema del equipo Innova para la Tarea 1 del shared task de SIMBig 2026: reconocimiento de voz para el quechua de Puno. Adelanto las tres cifras que estructuran la charla: mejoramos el baseline publicado de 13,15 a 9,14 de WER medio; ocho de esos puntos se explican solo por la ortografía de la referencia; y medimos que haber oído una frase en boca de otros hablantes vale 1,1 puntos.");

// ===================================================================== 2 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("El quechua de Puno y por qué ASR", { placeholder: "title" });
s.addText("Hablan con fluidez pero no escriben: la escolarización es en español. Por eso aquí el ASR no es una comodidad — una interfaz de texto excluye justo a quien debería servir.",
  { placeholder: "body", h: 0.6 });
cards(s, [
  { big: "465.000", label: "hablantes de quechua de Puno" },
  { big: "66 h", label: "de habla recolectada con diseño participativo\npor Huaman et al. (2026)" },
  { big: "4", label: "números componen el ranking: WER y CER en\nscripted y spontaneous" },
  { big: "659", label: "clips en el test oficial\n438 leídos · 221 espontáneos" },
], 2.5, 1.55);
s.addText("Promediar cuatro números descarta al mejor modelo del paper en scripted:",
  { x: M, y: 4.45, w: W, h: 0.3, isTextBox: true, margin: 0, fontSize: 14, color: INK, fontFace: THEME.bodyFontFace });
tbl(s, [
  [{ text: "Configuración del paper", options: { bold: true } }, { text: "scripted WER", options: { bold: true, align: "right" } }, { text: "spontaneous WER", options: { bold: true, align: "right" } }, { text: "Veredicto", options: { bold: true } }],
  [{ text: [code("xls-r + CPT"), txt(", solo validado")] }, { text: "1,19", options: { align: "right" } }, { text: "13,6", options: { align: "right", color: RUST } }, { text: "No: hunde dos de los cuatro", options: { color: MUTED } }],
  [{ text: [code("xls-r + CPT", { bold: true }), txt(", validado + silver", { bold: true })] }, { text: "2,11", options: { align: "right" } }, { text: "3,15", options: { align: "right" } }, { text: "Sí: equilibrado", options: { bold: true, color: GREEN } }],
], { y: 4.85, colW: [4.6, 2.0, 2.3, 2.73], rowH: 0.4 });
s.addNotes("El quechua de Puno lo hablan unas 465.000 personas, pero casi nadie lo escribe, porque la escolarización es en español. Por eso el ASR aquí no es una comodidad: una interfaz de texto deja fuera precisamente a la gente a la que queremos servir.\n\nDos cosas sobre la tarea. Partimos del corpus de 66 horas que publicaron los organizadores. Y el ranking promedia CUATRO números: WER y CER, en scripted y en spontaneous. Esto no es un detalle: descarta el modelo con mejor habla leída del paper, que tiene 1,19 de WER pero 13,6 en espontánea.");

// ===================================================================== 3 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Hallazgo 1: la ortografía vale casi ocho puntos", { placeholder: "title" });
s.addText(emph("El quechua de Puno no tiene ortografía estandarizada y los voluntarios de Common Voice escribieron como supieron. Los organizadores publicaron una referencia curada de las 2.067 frases: 629 cambian de grafía más allá de mayúsculas y puntuación.", { fontSize: 15, color: MUTED }), { placeholder: "body" });
cards(s, [
  { big: "allqu → alqu", label: "" },
  { big: "iskay → ishkay", label: "" },
  { big: "misk'i → mishk'i", label: "" },
], 2.5, 0.78, { bigSize: 19 });
s.addText("Un mismo conjunto de hipótesis del baseline, puntuado contra las dos referencias:",
  { x: M, y: 3.55, w: W, h: 0.3, isTextBox: true, margin: 0, fontSize: 14, color: INK, fontFace: THEME.bodyFontFace });
tbl(s, [
  [{ text: "Referencia", options: { bold: true } }, { text: "WER", options: { bold: true, align: "right" } }, { text: "CER", options: { bold: true, align: "right" } }],
  ["Texto crudo de Common Voice (con el que se entrenó el baseline)", { text: "8,13", options: { align: "right" } }, { text: "2,76", options: { align: "right" } }],
  [{ text: "Texto curado (contra el que puntúa el concurso)", options: { bold: true } }, { text: "16,08", options: { bold: true, align: "right", color: RUST } }, { text: "3,77", options: { bold: true, align: "right", color: RUST } }],
], { y: 3.95, colW: [8.2, 1.7, 1.73], rowH: 0.38 });
s.addText(emph("El modelo no cambió entre las dos medidas. Los 7,95 puntos de diferencia son ortografía, no acústica.", { fontSize: 14, color: RUST, bold: true }), { x: M, y: 5.6, w: W, h: 0.4, isTextBox: true, margin: 0, fontSize: 14, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addNotes("Primer hallazgo, y el más rentable. El quechua de Puno no tiene ortografía estándar, así que los voluntarios escribieron como supieron. Los organizadores publicaron una referencia curada: de las 2.067 frases, 629 cambian de grafía. Son cambios de una letra —allqu pasa a alqu, iskay a ishkay— pero en WER cada uno cuesta una palabra entera.\n\nEl experimento aísla el efecto. Cogimos el modelo del paper, le hicimos transcribir el dev UNA vez, y esas transcripciones congeladas las puntuamos contra las dos referencias. Contra el texto crudo da 8,13. Contra el curado, 16,08. El modelo es el mismo en ambas medidas, así que esos casi ocho puntos son ortografía pura.");

// ===================================================================== 4 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("El sistema", { placeholder: "title" });
s.addImage({
  path: path.join(__dirname, "..", "paper", "figures", "pipeline_pastel.png"),
  x: 3.09, y: 1.56, w: 7.16, h: 4.85, objectName: "pipeline",
});
s.addText(emph("Partimos del checkpoint publicado XLS-R-300M con pre-entrenamiento continuo, afinamos 500 pasos sobre el texto curado y decodificamos spontaneous con un modelo de lenguaje de caracteres.", { fontSize: 13, color: MUTED }), { x: M, y: 6.52, w: W, h: 0.5, isTextBox: true, margin: 0, fontSize: 13, color: MUTED, fontFace: THEME.bodyFontFace, align: "center" });
s.addNotes("Este es el sistema. De izquierda a derecha: los dos corpus más la referencia curada entran en la normalización, que lo pliega todo a un alfabeto de 29 símbolos.\n\nDe ahí salen DOS splits, y esto es lo importante. El de la izquierda separa por hablante, como hacía el baseline, pero comparte las frases entre train y dev. El de la derecha separa por hablante Y por texto.\n\nEl modelo que entregamos se entrena con el primero. Su hermano, con el segundo, y es el único del que tenemos una medición limpia. La flecha punteada marca lo que pasa de uno a otro: la receta, no los pesos.\n\nAbajo a la izquierda, la decodificación: habla leída con argmax voraz, espontánea con beam search y un modelo de lenguaje de caracteres de orden 8.");

// ===================================================================== 5 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Resultado principal", { placeholder: "title" });
s.addText("Sobre el split establecido, el mismo que usó el baseline, y contra la referencia curada. Todas las cifras en porcentaje.",
  { placeholder: "body", h: 0.45 });
tbl(s, [
  [{ text: "Sistema", options: { bold: true } },
   { text: "scripted WER", options: { bold: true, align: "right" } }, { text: "scripted CER", options: { bold: true, align: "right" } },
   { text: "spontaneous WER", options: { bold: true, align: "right" } }, { text: "spontaneous CER", options: { bold: true, align: "right" } },
   { text: "Media WER", options: { bold: true, align: "right" } }, { text: "Media CER", options: { bold: true, align: "right" } }],
  ["Baseline publicado, convertido", { text: "16,08", options: { align: "right" } }, { text: "3,77", options: { align: "right" } },
   { text: "10,23", options: { align: "right" } }, { text: "1,39", options: { align: "right" } },
   { text: "13,15", options: { align: "right" } }, { text: "2,58", options: { align: "right" } }],
  [{ text: "Este trabajo", options: { bold: true } }, { text: "9,23", options: { align: "right" } }, { text: "3,01", options: { align: "right" } },
   { text: "9,06", options: { align: "right" } }, { text: "1,30", options: { align: "right" } },
   { text: "9,14", options: { bold: true, align: "right", color: RUST } }, { text: "2,16", options: { align: "right" } }],
], { y: 2.3, colW: [4.0, 1.28, 1.28, 1.35, 1.35, 1.28, 1.09], rowH: 0.42 });
takeaways(s, [
  { head: "Comparación válida", body: "Mismo split, misma referencia y mismo modelo base que el baseline. Es una comparación de igual a igual." },
  { head: "Casi todo es ortografía", body: "La mejora de 6,85 puntos en scripted es del mismo orden que el desajuste de 7,95 que medimos antes." },
  { head: "Equilibrio entre dominios", body: "Mejora en los cuatro números a la vez, que es lo que exige el ranking por media." },
], 4.55);
s.addNotes("Este es el resultado. Mejoramos el baseline de 13,15 a 9,14 de WER medio, sobre el mismo split y con la misma referencia. Es una comparación de igual a igual: mismo modelo base, misma regla de medir.\n\nY quiero ser honesto sobre de dónde viene la mejora. En habla leída ganamos 6,85 puntos, que es del mismo orden que los 7,95 del desajuste ortográfico de la lámina anterior. Es decir: la mayor parte de lo que ganamos es ortografía, no acústica. Mejoramos en los cuatro números, que es lo que exige el ranking.");

// ===================================================================== 6 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Hallazgo 2: el test no reutiliza las frases del corpus", { placeholder: "title" });
s.addText(emph("Solo hay 2.067 frases y todas están grabadas, así que parecía seguro que el test las reutilizaría. Sin etiquetas: medimos la distancia de edición de cada hipótesis a la frase más cercana del corpus, calibrando con el dev.", { fontSize: 15, color: MUTED }), { placeholder: "body", h: 0.75 });
tbl(s, [
  [{ text: "Condición", options: { bold: true } }, { text: "Distancia mediana", options: { bold: true, align: "right" } }],
  ["Dev, con su frase presente en el corpus", { text: "0,000", options: { align: "right" } }],
  ["Dev, quitando su frase del corpus", { text: "0,516", options: { align: "right" } }],
  [{ text: [code("Test"), txt(", scripted", { bold: true })], options: { bold: true } }, { text: "0,480", options: { bold: true, align: "right", color: RUST } }],
], { y: 2.6, w: 7.5, colW: [5.1, 2.4], rowH: 0.42 });

s.addShape(pres.ShapeType.roundRect, { x: 8.9, y: 2.6, w: 3.6, h: 1.75, rectRadius: 0.06,
  fill: { color: CARD }, line: { color: CARD }, objectName: "card-test" });
s.addText(emph("0,480", { fontSize: 34, color: RUST }), { x: 9.18, y: 2.78, w: 3.1, h: 0.6, isTextBox: true, margin: 0,
  fontSize: 34, color: RUST, fontFace: THEME.headFontFace, valign: "middle" });
s.addText("la distancia del test: se comporta como el caso «frase quitada», no como «frase presente»",
  { x: 9.18, y: 3.42, w: 3.1, h: 0.8, isTextBox: true, margin: 0, fontSize: 12,
    color: MUTED, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.15 });

s.addText(emph("Sus frases son nuevas, y además más largas: de ocho a doce palabras frente a una media de cuatro en el corpus.", { fontSize: 14, color: INK }),
  { x: 8.9, y: 4.65, w: 3.6, h: 0.9, isTextBox: true, margin: 0, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.25 });

s.addText("Esto decide cómo se mide todo lo demás: un dev cuyas frases el modelo ha oído a otros once hablantes premia a los sistemas que recuerdan texto, no a los que reconocen habla.",
  { x: M, y: 5.5, w: 7.5, h: 0.9, isTextBox: true, margin: 0, fontSize: 15, bold: true, color: RUST, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.25 });

s.addNotes("Segundo hallazgo. El corpus tiene solo 2.067 frases y todas están grabadas, así que dábamos por hecho que el test las reutilizaría. Fuimos a comprobarlo.\n\nEl test llega sin transcripciones, así que la comprobación no puede usar etiquetas. Lo que hicimos: transcribir el audio del test con nuestro propio sistema y medir, para cada hipótesis, la distancia de edición a la frase más parecida del corpus. Calibramos con el dev, donde sí sabemos la respuesta.\n\nUna hipótesis del dev cuya frase está en el corpus queda a distancia cero. Si quitamos esa frase del corpus, sube a 0,516. El test da 0,480: se comporta como el caso de frase no vista. Son frases nuevas.");

// ===================================================================== 7 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Hallazgo 3: cuánto vale haber oído la frase", { placeholder: "title" });
s.addText(emph("El split establecido comparte todas sus frases de evaluación con el entrenamiento, y no por descuido: con 2.067 frases y 77 hablantes, no hay otra forma de usar las 22.727 grabaciones.", { fontSize: 15, color: MUTED }), { placeholder: "body", h: 0.65 });
tbl(s, [
  [{ text: "Entrenamiento", options: { bold: true } }, { text: "¿Oyó las frases de dev?", options: { bold: true } }, { text: "scripted WER", options: { bold: true, align: "right" } }],
  [{ text: [mvar("A"), txt("  ·  39,94 h")] }, "No", { text: "5,20  /  4,97", options: { align: "right" } }],
  [{ text: [mvar("B", { bold: true }), txt("eq", { bold: true, subscript: true }), txt("  ·  39,94 h", { bold: true })] },
   { text: [txt("Sí, con las mismas horas que ", { bold: true }), mvar("A", { bold: true })] },
   { text: "3,79  /  4,23", options: { bold: true, align: "right", color: RUST } }],
  [{ text: [mvar("B"), txt("  ·  43,58 h")] }, "Sí, con 3,65 h más de audio", { text: "4,68", options: { align: "right" } }],
], { y: 2.45, colW: [3.4, 5.2, 3.03], rowH: 0.4 });
s.addText([txt("Mismo dev y misma receta en los tres casos: lo único que cambia es qué vio el modelo. "), mvar("B"), txt("eq", { subscript: true }), txt(" quita 3,64 h al azar para igualar el volumen con "), mvar("A"), txt(", de modo que la única diferencia sea la exposición.")],
  { x: M, y: 4.55, w: W, h: 0.5, isTextBox: true, margin: 0, fontSize: 13, color: MUTED, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.2 });
cards(s, [
  { big: "1,1 puntos", label: "de WER, un 21 % relativo: los rangos de semillas de las dos condiciones no se solapan" },
  { big: "Cota inferior", label: "la memorización crece con el entrenamiento, y el baseline entrena 8.000 pasos" },
], 5.35, 1.3, { bigSize: 23 });
s.addNotes("Tercer hallazgo, y el que creo que sobrevivirá al sistema.\n\nEl split establecido comparte TODAS las frases de evaluación con el entrenamiento. No es un descuido: con 2.067 frases y 77 hablantes, si quieres usar las 22.000 grabaciones, necesitas las 2.067 frases en train. Es una propiedad del benchmark.\n\nPara ponerle número diseñamos un experimento controlado. El dev es el mismo en los tres casos y la receta también; lo único que cambia es qué vio el modelo. A no oye las frases de dev. B sí las oye, con 3,65 horas más de audio. Y B_eq las oye pero con las mismas horas que A, quitando clips al azar, para que la única diferencia sea la exposición.\n\nLos rangos de semillas de A y B_eq no se solapan: el peor B_eq sigue siendo mejor que el mejor A. Haber oído la frase en boca de otros vale 1,1 puntos de WER, un 21 % relativo. Y es una cota inferior, porque la memorización crece con el entrenamiento y el baseline entrena 8.000 pasos.");

// ===================================================================== 8 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Cuántos pasos entrenar, y cuánto es ruido", { placeholder: "title" });
s.addText(emph("Partiendo de un checkpoint ya adaptado hace falta mucho menos entrenamiento que los 8.000 pasos de la receta original. Entre 250 y 1.000 pasos, y con tasas de 1e-5 a 5e-5, todo cae entre 7,1 y 7,3 de WER medio.", { fontSize: 15, color: MUTED }), { placeholder: "body", h: 0.75 });
cards(s, [
  { big: "7,02", label: "WER medio con el dev disjunto en texto\n(339 clips): desviación 0,19" },
  { big: "9,23", label: "WER medio con el dev por hablante\n(1.493 clips): desviación 0,08" },
  { big: "0,3", label: "umbral de ruido: por debajo, una diferencia\nno se distingue del azar" },
], 2.45, 1.5);
s.addText([
  { text: "La corrección que tuvimos que hacer.  ", options: { bold: true, color: RUST } },
  { text: "Una corrida de 500 pasos a 2e-5 dio 6,96 de WER medio, la mejor de todas las configuraciones en los cuatro números, y la anotamos como ganadora. Una segunda semilla de la receta idéntica dio 7,26. Era suerte.", options: { color: INK } },
], { x: M, y: 4.3, w: W, h: 0.9, isTextBox: true, margin: 0, fontSize: 14, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.25 });
s.addText("El dev pequeño no solo mide con menos precisión: fabrica diferencias. Varias de nuestras conclusiones tempranas no sobrevivieron a una segunda semilla.",
  { x: M, y: 5.35, w: W, h: 0.5, isTextBox: true, margin: 0, fontSize: 14, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addNotes("Sobre cuánto entrenar. Partiendo de un checkpoint ya adaptado al quechua, hacen falta muchos menos pasos que los 8.000 de la receta original. Entre 250 y 1.000 pasos, con tasas de aprendizaje de 1e-5 a 5e-5, todo cae entre 7,1 y 7,3. Es una meseta ancha, no un óptimo.\n\nY aquí una corrección que creo que vale la pena contar. Una corrida de 500 pasos con tasa 2e-5 dio 6,96 de media, la mejor de todas las configuraciones en los cuatro números, y la anotamos como ganadora. Una segunda semilla de la receta IDÉNTICA dio 7,26. Era suerte.\n\nPor eso medimos el ruido con tres semillas y fijamos un umbral de 0,3: por debajo de eso, una diferencia no se distingue del azar. Y fíjense en el contraste entre los dos dev: el pequeño tiene más del doble de dispersión. Un dev pequeño no solo mide peor, fabrica diferencias.");

// ===================================================================== 9 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Lo que no funcionó: promediar modelos", { placeholder: "title" });
s.addText(emph("Un ruido de 0,3 entre semillas es justo lo que el ensemble debería eliminar, así que esperábamos una ganancia. Ninguna de las dos formas de combinar la da, y el fallo tiene un mecanismo que pudimos medir.", { fontSize: 15, color: MUTED }), { placeholder: "body", h: 0.75 });
tbl(s, [
  [{ text: "Sistema", options: { bold: true } }, { text: "Media WER", options: { bold: true, align: "right" } }, { text: "Media CER", options: { bold: true, align: "right" } }],
  [{ text: "Media de las tres semillas por separado", options: { bold: true } }, { text: "7,02", options: { bold: true, align: "right" } }, { text: "0,96", options: { bold: true, align: "right" } }],
  ["Ensemble promediando logits", { text: "8,38", options: { align: "right", color: RUST } }, { text: "1,14", options: { align: "right", color: RUST } }],
  ["Ensemble promediando probabilidades", { text: "7,30", options: { align: "right" } }, { text: "1,02", options: { align: "right" } }],
], { y: 2.45, w: 7.4, colW: [4.3, 1.6, 1.5], rowH: 0.4 });
s.addText([
  { text: "El mecanismo\n", options: { bold: true, color: RUST, breakLine: true } },
  { text: "CTC no emite letras de forma continua: emite picos agudos en frames concretos. Modelos entrenados por separado colocan esos picos en frames contiguos pero no idénticos.\n\n", options: { breakLine: true } },
  { text: "Promediar logits es una media geométrica, así que un pico seguro promediado contra un valor casi nulo se desploma y la letra desaparece: los borrados de carácter suben de 21 a 50.", options: {} },
], { x: 8.5, y: 2.45, w: 4.0, h: 2.6, isTextBox: true, margin: 0, fontSize: 12, color: INK, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.25 });
s.addText("Con las alineaciones de CTC, agudas y desplazadas en el tiempo, la regla de fusión importa más que el ensemble.",
  { x: M, y: 5.4, w: W, h: 0.4, isTextBox: true, margin: 0, fontSize: 14, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addNotes("Tres cosas que esperábamos que ayudaran y no lo hicieron. La primera: promediar modelos entre semillas.\n\nEl ruido de 0,3 entre semillas es exactamente lo que un ensemble debería eliminar, así que esperábamos ganar. No ganamos con ninguna de las dos formas de combinar.\n\nY el fallo tiene un mecanismo que pudimos medir. CTC no emite letras de forma continua: emite picos agudos en frames concretos, y blancos en el resto. Modelos entrenados por separado colocan esos picos en frames contiguos pero no idénticos. Promediar logits es una media geométrica, así que un pico seguro promediado contra un valor casi nulo se desploma, y la letra desaparece. Lo contamos: los borrados de carácter suben de 21 a 50.\n\nPromediando probabilidades, que es media aritmética, ya no se pierden letras, pero tampoco se gana nada. La lección: con CTC, la regla de fusión importa más que el ensemble.");

// ==================================================================== 10 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Lo que no funcionó: máscara y modelo de lenguaje", { placeholder: "title" });
s.addText("Más regularización",
  { x: M, y: 1.45, w: 5.6, h: 0.3, isTextBox: true, margin: 0, fontSize: 16, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addText(emph("El sobreajuste pasados los 1.000 pasos era el fallo que observamos, y SpecAugment lo ataca directamente. Subir la probabilidad de enmascarado desde 0,05 empeora de forma monótona, y en los cuatro números.", { fontSize: 13, color: MUTED }), { x: M, y: 1.82, w: 5.6, h: 1.0, isTextBox: true, margin: 0, fontSize: 13, color: MUTED, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.2 });
tbl(s, [
  [{ text: [code("mask_time_prob", { bold: true })] }, { text: "Media WER", options: { bold: true, align: "right" } }],
  [{ text: "0,05  (por defecto)", options: { bold: true } }, { text: "6,81", options: { bold: true, align: "right" } }],
  ["0,20", { text: "7,05", options: { align: "right" } }],
  ["0,50", { text: "7,68", options: { align: "right", color: RUST } }],
], { x: M, y: 2.95, w: 5.6, colW: [3.5, 2.1], rowH: 0.38 });
s.addText("El modelo de lenguaje deja de ayudar",
  { x: 7.1, y: 1.45, w: 5.4, h: 0.3, isTextBox: true, margin: 0, fontSize: 16, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addText(emph("En habla leída ninguna configuración gana de forma medible: todas quedan a ±0,15 de la decodificación voraz, unas dos palabras. Por eso el sistema final decodifica scripted sin modelo de lenguaje.", { fontSize: 13, color: MUTED }), { x: 7.1, y: 1.82, w: 5.4, h: 1.0, isTextBox: true, margin: 0, fontSize: 13, color: MUTED, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.2 });
tbl(s, [
  [{ text: "Semilla", options: { bold: true } }, { text: "Dev", options: { bold: true, align: "right" } }, { text: "Heldout", options: { bold: true, align: "right" } }],
  ["42", { text: "+0,31", options: { align: "right" } }, { text: "0,00", options: { align: "right" } }],
  ["43", { text: "+0,17", options: { align: "right" } }, { text: "−0,04", options: { align: "right", color: RUST } }],
  ["44", { text: "+0,12", options: { align: "right" } }, { text: "−0,08", options: { align: "right", color: RUST } }],
], { x: 7.1, y: 2.95, w: 5.4, colW: [2.0, 1.7, 1.7], rowH: 0.38 });
s.addText(emph("Ganancia en WER de spontaneous frente a decodificación voraz. En el dev gana entre 0,12 y 0,31; en el conjunto reservado, nada.", { fontSize: 12, color: MUTED }), { x: 7.1, y: 4.85, w: 5.4, h: 0.6, isTextBox: true, margin: 0, fontSize: 12, color: MUTED, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.2 });
s.addText("Una ganancia ajustada sobre el dev no sobrevivió a un conjunto limpio, aun habiéndose ajustado sobre el split disjunto en texto.",
  { x: M, y: 5.75, w: W, h: 0.5, isTextBox: true, margin: 0, fontSize: 14, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addNotes("Las otras dos cosas que no funcionaron.\n\nA la izquierda, la regularización. El sobreajuste pasados los mil pasos era el fallo que observamos, y SpecAugment lo ataca directamente. Subir el enmascarado empeora de forma monótona: 6,81, 7,05, 7,68. Y empeora en los cuatro números. Con una receta de 500 pasos hay poco tiempo para sobreajustar, y la máscara quita evidencia que el modelo tiene pocas ocasiones de ver.\n\nA la derecha, el modelo de lenguaje. En habla leída ninguna configuración gana de forma medible, así que el sistema final decodifica scripted sin él. En espontánea sí gana en el dev, entre 0,12 y 0,31 según la semilla. Pero en el conjunto reservado gana cero y pierde 0,04 y 0,08.\n\nEs decir: una ganancia ajustada sobre el dev no sobrevivió a un conjunto limpio, y eso que la habíamos ajustado sobre el split disjunto en texto. Lo mantenemos porque el efecto está dentro del ruido en ambas direcciones, pero ya no afirmamos que ayude.");

// ==================================================================== 11 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("El conjunto reservado, medido una sola vez", { placeholder: "title" });
s.addText(emph("Un dev que ha guiado todas las decisiones deja de ser imparcial. El nuestro guió pasos, tasas, 72 configuraciones de modelo de lenguaje y la decodificación por dominio: nada de eso es entrenamiento, pero todo es selección.", { fontSize: 15, color: MUTED }), { placeholder: "body", h: 0.7 });
tbl(s, [
  ["", { text: "scripted WER", options: { bold: true, align: "right" } }, { text: "scripted CER", options: { bold: true, align: "right" } },
   { text: "spontaneous WER", options: { bold: true, align: "right" } }, { text: "spontaneous CER", options: { bold: true, align: "right" } },
   { text: "Media WER", options: { bold: true, align: "right" } }],
  [{ text: "Desarrollo", options: { bold: true } }, { text: "4,85", options: { align: "right" } }, { text: "0,62", options: { align: "right" } },
   { text: "9,20", options: { align: "right" } }, { text: "1,31", options: { align: "right" } }, { text: "7,02", options: { align: "right" } }],
  [{ text: "Reservado", options: { bold: true } }, { text: "10,79", options: { align: "right" } }, { text: "2,33", options: { align: "right" } },
   { text: "3,38", options: { align: "right" } }, { text: "0,39", options: { align: "right" } }, { text: "7,08", options: { bold: true, align: "right", color: GREEN } }],
], { y: 2.75, w: 11.0, colW: [2.6, 1.68, 1.68, 1.68, 1.68, 1.68], rowH: 0.42 });
takeaways(s, [
  { head: "Las medias coinciden", body: "7,02 frente a 7,08. Después de decenas de decisiones tomadas sobre el dev, la estimación limpia no se desploma." },
  { head: "Pero esconden dos movimientos opuestos", body: "La habla leída dobla su error y la espontánea lo divide por tres. Es composición, no generalización." },
  { head: "Un hablante lo explica casi todo", body: "El reservado incluye a uno que él solo da 98,67 de WER sobre 59 clips." },
], 4.85);
s.addNotes("Esta lámina es la que más me importa metodológicamente.\n\nUn dev que ha guiado todas las decisiones deja de ser una estimación imparcial de nada. El nuestro guió el número de pasos, las tasas, 72 configuraciones de modelo de lenguaje y qué decodificador usar en cada dominio. Nada de eso es entrenamiento en sentido estricto, pero todo es selección.\n\nPor eso apartamos un conjunto el primer día, no lo miramos nunca, y lo medimos UNA sola vez, después de congelar todas las decisiones. Para que el compromiso fuera comprobable y no solo declarado, pre-registramos las cifras esperadas en la cabecera del script antes de ejecutarlo.\n\nLas medias coinciden: 7,02 contra 7,08. Después de decenas de decisiones tomadas sobre el dev, la estimación limpia no se desploma. Pero esa coincidencia esconde dos movimientos grandes y opuestos: la habla leída dobla su error y la espontánea lo divide por tres. Y ambos son composición, no generalización: el reservado contiene un hablante que él solo da 98,67 de WER sobre 59 clips.");

// ==================================================================== 12 ====
s = pres.addSlide({ masterName: "CONTENIDO" });
s.addText("Limitaciones", { placeholder: "title" });
cards(s, [
  { big: "1", label: "sola transcripción humana de hablantes de 60 o más años en todo el corpus — y el test espontáneo son íntegramente hablantes de esa edad" },
  { big: "2 de 9", label: "hiperparámetros variados: solo pasos y tasa. Lote, upsampling, layer dropout, acentos y datos silver quedaron por defecto" },
  { big: "3", label: "semillas: bastan para tumbar victorias aparentes, no para separar recetas que difieren menos que su dispersión" },
], 1.85, 2.05, { bigSize: 30 });

s.addText("Y la que más pesa: el sistema entregado no tiene medición limpia propia",
  { x: M, y: 4.35, w: W, h: 0.35, isTextBox: true, margin: 0, fontSize: 17, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addText(emph("Se entrena sobre el split establecido, cuyo train contiene las frases de dev. Lo entregamos por comparabilidad con el baseline publicado y porque ve un 15 % más de frases distintas — pero eso es una apuesta razonada, no medida. El efecto de 1,1 puntos de la lámina 7 mide memorización y no sirve para justificarla.", { fontSize: 14, color: INK }), { x: M, y: 4.78, w: W, h: 1.0, isTextBox: true, margin: 0, fontSize: 14, color: INK, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.25 });

s.addNotes("Cuatro limitaciones, y las digo porque creo que declararlas fortalece el trabajo.\n\nLa primera, la más incómoda: el sistema que entregamos no tiene una medición limpia propia. Se entrena sobre el split establecido, cuyo train contiene las frases de dev. Lo entregamos por comparabilidad con el baseline publicado y porque ve un quince por ciento más de frases distintas. Pero eso último es una apuesta razonada, no medida, y el efecto de 1,1 puntos que medimos antes es memorización y no sirve para justificarla.\n\nSegunda: el barrido fue estrecho. Solo pasos y tasa de aprendizaje.\n\nTercera: tres semillas bastan para tumbar victorias aparentes, pero no para separar recetas parecidas.\n\nY la cuarta, la más seria: la mitad del ranking no la podemos medir con fidelidad. El test espontáneo son íntegramente hablantes de sesenta años o más, y en todo el corpus esos hablantes aportan UNA transcripción humana. El dominio que decide la mitad de nuestro puesto es el que menos podemos evaluar.");

// ==================================================================== 13 ====
s = pres.addSlide({ masterName: "CIERRE" });
s.addText("Lo que nos llevamos", { placeholder: "title" });
cards(s, [
  { big: "13,15 → 9,14", label: "de WER medio frente al baseline publicado,\nsobre el mismo split y la misma referencia.\nCasi todo es ortografía, no acústica." },
  { big: "1,1 puntos", label: "es lo que vale el solapamiento de frases\nen el benchmark establecido, medido\ncon un experimento controlado." },
  { big: "3 resultados", label: "negativos reproducibles: el ensemble de\nlogits daña CTC, más máscara perjudica,\ny la ganancia del LM no sobrevive." },
], 2.1, 1.95, { bigSize: 25 });
s.addText("Dos direcciones a futuro",
  { x: M, y: 4.5, w: W, h: 0.3, isTextBox: true, margin: 0, fontSize: 16, bold: true, color: RUST, fontFace: THEME.bodyFontFace });
s.addText(emph("Un dev disjunto en texto construido a partir del split oficial de Common Voice daría 679 clips honestos en vez de 339, a cambio de seis horas de audio de entrenamiento. Y, más importante, transcripciones humanas de hablantes mayores harían medible por primera vez la mitad de este benchmark.", { fontSize: 14, color: INK }), { x: M, y: 4.88, w: W, h: 0.8, isTextBox: true, margin: 0, fontSize: 14, color: INK, fontFace: THEME.bodyFontFace, lineSpacingMultiple: 1.25 });
s.addText([txt("Código, splits y configuraciones:  "), code("github.com/raymond1242/asr-puno-quechua")],
  { x: M, y: 6.1, w: W, h: 0.3, isTextBox: true, margin: 0, fontSize: 12, color: MUTED, fontFace: THEME.bodyFontFace });
s.addNotes("Para cerrar, tres cosas.\n\nMejoramos el baseline publicado de 13,15 a 9,14 de WER medio, sobre el mismo split y con la misma referencia. Y somos explícitos en que casi todo ese avance es ortografía, no acústica.\n\nSegundo, y creo que es lo que más puede durar: pusimos número al solapamiento de frases del benchmark establecido. Vale 1,1 puntos, medidos con un experimento controlado, y mostramos que el test del propio concurso no comparte esas frases.\n\nY tercero, tres resultados negativos reproducibles, con el mecanismo explicado en el caso del ensemble.\n\nDos direcciones a futuro. Un dev disjunto en texto construido sobre el split oficial de Common Voice daría el doble de clips honestos. Y, sobre todo, transcripciones humanas de hablantes mayores: harían medible por primera vez la mitad de este benchmark.\n\nTodo el código y los splits están en el repositorio. Muchas gracias.");

(async () => {
  const out = path.join(__dirname, "simbig2026_task1.pptx");
  await pres.writeFile({ fileName: out });
  const { applyTheme } = require(path.join(
    process.env.PPTX_SKILL || "/Users/raymondnegron/.claude/skills/synced/d8c2bb87-ac0b-454c-a267-0537a651d187_90044e1a-bd1d-4be9-a2c4-9690216c2444/pptx",
    "scripts", "apply_theme.js"));
  await applyTheme(out, THEME);
  console.log("ok ->", out);
})();
