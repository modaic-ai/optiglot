import * as fs from "node:fs";
import * as path from "node:path";
import { spawn } from "node:child_process";
import { Module } from "./module";
import { Dataset } from "./dataset";
import { createServer } from "./server";
import { type MetricFunction } from "./models";

export class Teleprompt {
  optimizerName: string;
  options: any;
  metric?: MetricFunction;
  dataset?: Dataset;
  private finalizeResolve?: (result: any) => void;

  constructor(optimizerName: string, options: any = {}) {
    this.optimizerName = optimizerName;
    this.options = {
      numThreads: 4, // Default value
      ...options,
    };
  }

  async finalize(bestCandidate: Record<string, string>, results: any) {
    if (this.finalizeResolve) {
      this.finalizeResolve({ bestCandidate, results });
    }
  }

  async compile(
    student: Module,
    metric: MetricFunction,
    dataset: Dataset,
    valset?: Dataset
  ): Promise<Module> {
    this.metric = metric;
    this.dataset = dataset;
    const server = createServer(this, student);
    const port = 8000;

    await server.listen({ port, host: "0.0.0.0" });
    console.log(
      `${this.optimizerName} optimization server started on port ${port}`
    );

    // Create a temporary directory for optimization artifacts
    const logDir =
      this.options.log_dir || path.join(process.cwd(), ".optiglot_run");
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    try {
      // 1. Prepare dataset path
      const datasetPath = dataset.toFile(logDir);
      const valsetPath = valset ? valset.toFile(logDir) : datasetPath;

      // 2. Prepare seed candidate from student predictors
      const seedCandidate: Record<string, string> = {};
      for (const [name, predictor] of Object.entries(student._predictors)) {
        seedCandidate[name] = predictor.instructions;
      }

      // 3. Generate config.yaml (using JSON format as it is valid YAML)
      const configPath = path.join(logDir, "config.yaml");
      const { numThreads, ...restOptions } = this.options;
      const config: Record<string, any> = {
        ...restOptions,
        optimizer: this.optimizerName,
        trainset: datasetPath,
        valset: valsetPath,
        seed_candidate: seedCandidate,
        adapter: {
          base_url: `http://localhost:${port}`,
        },
      };

      fs.writeFileSync(configPath, JSON.stringify(config, null, 2));

      // 4. Spawn CLI
      console.log(`Starting ${this.optimizerName} optimization process...`);

      const optimizationPromise = new Promise<{
        bestCandidate: Record<string, string>;
        results: any;
      }>((resolve) => {
        this.finalizeResolve = resolve;
      });

      const isDev = process.env.OPTIGLOT_DEV === "true";
      const cmd = isDev ? "uv" : "uvx";
      const args = isDev
        ? ["run", "gepa-rpc", "--port", String(port), "--config", configPath]
        : ["gepa-rpc", "--port", String(port), "--config", configPath];

      const pythonProcess = spawn(cmd, args, {
        stdio: "inherit",
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      return await new Promise((resolve, reject) => {
        pythonProcess.on("error", (err) => {
          console.error("Failed to start optimization process:", err);
          reject(err);
        });

        optimizationPromise.then(({ bestCandidate, results }) => {
          // Update student with best candidate
          for (const [name, prompt] of Object.entries(bestCandidate)) {
            if (student._predictors[name]) {
              student._predictors[name].instructions = prompt;
            }
          }
          console.log("Optimization completed via /finalize callback.");
          resolve(student);
        });

        pythonProcess.on("close", (code) => {
          if (code !== 0) {
            reject(new Error(`Optimization process failed with code ${code}`));
          }
        });
      });
    } catch (error) {
      console.error("Optimization failed:", error);
      throw error;
    } finally {
      await server.close();
      console.log(`${this.optimizerName} optimization server stopped`);
    }
  }
}
