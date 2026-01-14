import Fastify from "fastify";
import type { EvaluateRequest, FinalizeRequest, MetricRequest } from "./models";
import { Teleprompt } from "./teleprompt";
import { Module } from "./module";

export function createServer(teleprompt: Teleprompt, module: Module) {
  const fastify = Fastify({
    connectionTimeout: 0, // time to receive full request
    keepAliveTimeout: 0, // keep-alive between requests
  });

  fastify.post("/evaluate", async (request, reply) => {
    request.raw.setTimeout(0);
    reply.raw.setTimeout(0);

    const { batch, candidate, capture_traces } =
      request.body as EvaluateRequest;

    if (!teleprompt.metric) {
      reply
        .status(500)
        .send({ error: "Metric not configured on Teleprompt instance" });
      return;
    }

    const response = await module.evaluate(
      batch,
      candidate,
      capture_traces,
      teleprompt.options.numThreads || 4,
      teleprompt.metric,
      teleprompt.dataset
        ? teleprompt.dataset.getInputs.bind(teleprompt.dataset)
        : undefined
    );

    return response;
  });

  fastify.post("/metric", async (request, reply) => {
    const { entries } = request.body as MetricRequest;

    if (!teleprompt.metric) {
      reply
        .status(500)
        .send({ error: "Metric not configured on Teleprompt instance" });
      return;
    }

    const scores = await Promise.all(
      entries.map(async (entry) => {
        const metricVal = await teleprompt.metric!(
          entry.gold,
          entry.prediction,
          entry.trace,
          entry.pred_name,
          entry.pred_trace
        );

        if (typeof metricVal === "number") {
          return {
            score: metricVal,
            feedback: `This trajectory got a score of ${metricVal}.`,
          };
        } else {
          return {
            score: metricVal.score,
            feedback:
              metricVal.feedback ??
              `This trajectory got a score of ${metricVal.score}.`,
          };
        }
      })
    );

    return { scores };
  });

  fastify.post("/finalize", async (request, reply) => {
    const { best_candidate, results } = request.body as FinalizeRequest;
    await teleprompt.finalize(best_candidate, results);
    return { status: "ok" };
  });

  return fastify;
}
