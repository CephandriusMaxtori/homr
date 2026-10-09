/**
 * Full end-to-end OMR pipeline orchestrator for homr-web.
 */

import type { InferenceSession } from "onnxruntime-web";
import { getCv } from "./cvwrap.ts";
import { generateXml, XmlGeneratorArguments, xmlToString } from "./music_xml_generator.ts";
import { runSegnetInference } from "./segnet.ts";
import { findHorizontalLinesCv, Staff, type StaffPoint } from "./staff_detection.ts";
import { dewarpStaffCanvas } from "./staff_dewarping.ts";
import { runEncoder, runGreedyDecoder } from "./tromr.ts";
import type { EncodedSymbol } from "./vocabulary.ts";

export interface PipelineProgress {
  stage: string;
  percent: number;
}

export interface PipelineOptions {
  onProgress?: (progress: PipelineProgress) => void;
  precision?: "fp16" | "fp32";
}

export interface ModelSessions {
  segnet: InferenceSession;
  encoder: InferenceSession;
  decoder: InferenceSession;
}

export async function processImagePipeline(
  models: ModelSessions,
  imageGray: Uint8Array,
  width: number,
  height: number,
  options: PipelineOptions = {},
): Promise<string> {
  const precision = options.precision ?? "fp32";

  options.onProgress?.({ stage: "Running segmentation model...", percent: 10 });
  const segnetLayers = await runSegnetInference(
    models.segnet,
    imageGray,
    width,
    height,
    precision,
  );

  options.onProgress?.({ stage: "Detecting staff lines...", percent: 40 });
  const staffLineGroups = await findHorizontalLinesCv(segnetLayers.staff, width, height);

  options.onProgress?.({ stage: "Constructing staff grids...", percent: 50 });
  const staves: Staff[] = staffLineGroups.map((group) => {
    const grid: StaffPoint[] = [];
    for (let x = 0; x < width; x += 10) {
      grid.push({ x, y: group, angle: 0 });
    }
    return new Staff(grid);
  });

  const recognizedStaves: EncodedSymbol[][] = [];
  const cvInstance = await getCv();

  for (let idx = 0; idx < staves.length; idx++) {
    const progressPercent = 50 + Math.round(((idx + 1) / staves.length) * 40);
    options.onProgress?.({
      stage: `Recognizing staff ${idx + 1} of ${staves.length}...`,
      percent: progressPercent,
    });

    const staff = staves[idx];
    if (!staff) continue;

    const dewarpedCanvas = await dewarpStaffCanvas(
      cvInstance,
      imageGray,
      width,
      height,
      [[{ x: 0, y: staff.minY }, { x: width, y: staff.minY }]],
      [[{ x: 0, y: staff.minY }, { x: width, y: staff.minY }]],
    );

    const context = await runEncoder(models.encoder, dewarpedCanvas, width, height, precision);
    const symbols = await runGreedyDecoder(models.decoder, context, precision);
    recognizedStaves.push(symbols);
  }

  options.onProgress?.({ stage: "Generating MusicXML...", percent: 95 });
  const xmlElement = generateXml(new XmlGeneratorArguments(), recognizedStaves, "homr Web Score");
  const resultXml = xmlToString(xmlElement);

  options.onProgress?.({ stage: "Complete", percent: 100 });
  return resultXml;
}
