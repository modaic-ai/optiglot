import fs from "fs";
import { nanoid } from "nanoid";
import { AsyncLocalStorage } from "async_hooks";
import type {
  Predict as PredictInterface,
  TraceEntry,
  Trace,
  Prediction,
  MetricFunction,
  EvaluationBatch,
  Example,
  ScoreWFeedback,
} from "./models";

export const requestContext = new AsyncLocalStorage<{
  runId: string;
  trace: TraceEntry<any, any>[];
}>();

type Predict = PredictInterface & { name: string };

interface RunResult {
  prediction: Prediction<any>;
  trace: TraceEntry<any, any>[];
}

export class Module<Input = any, Output = any> {
  [key: string]: any;
  private _forward?: (input: Input) => Promise<Output>;
  _predictors: Record<string, Predict>;

  constructor(
    predictors: Record<string, Predict>,
    forward?: (input: Input) => Promise<Output>
  ) {
    this._predictors = predictors;
    for (const [name, predictor] of Object.entries(predictors)) {
      predictor.name = name;
      this[name] = predictor;
    }
    this._forward = forward;
  }

  clone(): Module<Input, Output> {
    const clonedPredictors = JSON.parse(JSON.stringify(this._predictors));
    return new Module(clonedPredictors, this.forward);
  }

  save(filePath: string): void {
    const state: Record<string, string> = {};
    for (const [name, predictor] of Object.entries(this._predictors)) {
      state[name] = predictor.instructions;
    }
    fs.writeFileSync(filePath, JSON.stringify(state, null, 2));
  }

  load(filePath: string): void {
    const content = fs.readFileSync(filePath, "utf-8");
    const state = JSON.parse(content);
    for (const [name, prompt] of Object.entries(state)) {
      if (this._predictors[name]) {
        this._predictors[name].instructions = prompt as string;
      }
    }
  }

  setForward(forward: (inputs: Input) => Promise<Output>) {
    this._forward = forward;
  }

  async forward(inputs: Input): Promise<Output> {
    if (!this._forward) {
      throw new Error("Forward function is not set");
    }
    return await this._forward(inputs);
  }

  async run(inputs: Input): Promise<RunResult> {
    const trace: TraceEntry<any, any>[] = [];
    let prediction: Prediction<any>;

    try {
      const output = await requestContext.run(
        { runId: nanoid(), trace },
        async () => {
          return await this.forward!(inputs);
        }
      );
      prediction = {
        output,
        errorType: null,
        errorMessage: null,
        errorTraceback: null,
      };
    } catch (e: any) {
      prediction = {
        output: null,
        errorType: e.name || "Error",
        errorMessage: e.message || String(e),
        errorTraceback: e.stack || null,
      };
    }

    return {
      prediction,
      trace,
    };
  }

  async evaluate(
    batch: Example[],
    candidate: Record<string, string>,
    capture_traces: boolean,
    numWorkers: number,
    metric: MetricFunction,
    getInputs?: (ex: Record<string, any>) => Record<string, any>
  ): Promise<EvaluationBatch<Trace<any, any>, Prediction<any>>> {
    // 1. Update candidate prompts
    for (const [name, prompt] of Object.entries(candidate)) {
      if (this[name] && typeof this[name] !== "function") {
        (this[name] as any).instructions = prompt;
      }
    }

    // 2. Run batch with worker pool
    const outputs: Prediction<any>[] = new Array(batch.length);
    const scores: number[] = new Array(batch.length);
    const trajectories: Trace<any, any>[] = new Array(batch.length);

    let currentIndex = 0;
    const workers = Array.from({ length: numWorkers }).map(async () => {
      while (currentIndex < batch.length) {
        const index = currentIndex++;
        if (index >= batch.length) break;

        const example = batch[index]!;
        const inputs = getInputs ? getInputs(example) : example;
        const runResult = await this.run(inputs as Input);

        let metricResult: ScoreWFeedback;
        try {
          const metricVal = await metric(
            example,
            runResult.prediction,
            runResult.trace
          );
          if (typeof metricVal === "number") {
            metricResult = {
              score: metricVal,
              feedback: `This trajectory got a score of ${metricVal}.`,
            };
          } else {
            metricResult = {
              score: metricVal.score,
              feedback:
                metricVal.feedback ??
                `This trajectory got a score of ${metricVal.score}.`,
            };
          }
        } catch (e) {
          console.error(`Metric evaluation failed for example ${index}:`, e);
          metricResult = { score: 0, feedback: "Metric evaluation failed." };
        }

        outputs[index] = runResult.prediction;
        scores[index] = metricResult.score;

        if (capture_traces) {
          trajectories[index] = {
            example_ind: index,
            example: example,
            prediction: runResult.prediction,
            trace: runResult.trace,
            score: {
              score: metricResult.score,
              feedback: metricResult.feedback,
            },
          };
        }
      }
    });

    await Promise.all(workers);

    return {
      outputs,
      scores,
      trajectories: capture_traces ? trajectories : ([] as Trace<any, any>[]),
    };
  }
}
