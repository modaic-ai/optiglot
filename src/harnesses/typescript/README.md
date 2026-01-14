# Optiglot

`optiglot` is a language-agnostic interface for prompt optimization engines (like **DSPy**). It allows you to use powerful optimization algorithms implemented in Python with your language/framework of choice via a standardized RPC protocol. By separating the optimization logic from the execution environment, `optiglot` makes prompt optimizers "polyglots" hence the name.

## Installation

### TypeScript/JavaScript

Install the harness in your project:

```bash
npm install optiglot
# or
bun add optiglot
```

### CLI Engine

Install the optimization engine via `uv`:

```bash
uv tool install optiglot
```

---

## Core Concepts

- **Harnesses**: The "glue" that connects your framework (e.g., Vercel AI SDK) to the optimization engine. They handle tracing, evaluation, and communication.
- **Predictors**: Wrappers around LLM calls that record inputs/outputs for the optimizer.
- **Module**: A collection of predictors that define your AI system's logic.
- **Teleprompt**: The optimizer that runs the optimization loop.

---

## Defining an Optimizer

In Optiglot, an optimizer is defined using the `Teleprompt` class. You specify which algorithm to use and its configuration.

```typescript
import { Teleprompt } from "optiglot";

const optimizer = new Teleprompt("bootstrap", {
  numThreads: 4, // Number of concurrent evaluations
  maxIterations: 10, // Maximum optimization cycles
  // ... other algorithm-specific options
});
```

Common optimizers include:

- `vanilla`: Basic few-shot prompting.
- `bootstrap`: Bootstraps few-shot examples from your training data.
- `teleprompt`: Advanced multi-stage optimization.

---

## The Purpose of Harnesses

Harnesses are essential because they bridge the gap between your application code and the optimization engine. They abstract away three critical tasks:

1. **Tracing**: Automatically recording LLM inputs and outputs during a run so the optimizer can "see" what happened.
2. **Execution Control**: Allowing the optimizer to swap out prompts and re-run your code during evaluation.
3. **Communication**: Handling the RPC handshake between your application and the Python-based optimization engine.

By using a harness, you can optimize your prompts without rewriting your existing application logic in Python.

---

## Usage: AI SDK Harness

Here's how to use the Vercel AI SDK harness to optimize a support ticket classifier.

### 1. Define Your System (`Module`)

Create a `Module` with your predictors. A `Predict` component wraps your LLM call and manages its instructions.

```typescript
import { Module, Predict } from "optiglot";
import { openai } from "@ai-sdk/openai";

// Define the system structure
const classifier = new Module({
  judge: new Predict(
    "Classify the ticket into: Billing, Technical, or General."
  ),
});

// Define the execution logic
classifier.setForward(async (inputs: { ticket: string }) => {
  const { text } = await classifier.judge.generateText({
    model: openai("gpt-4o-mini"),
    prompt: `Ticket: ${inputs.ticket}`,
  });
  return text;
});
```

### 2. Define Your Metric

The metric scores performance on a specific example. It can return a simple score (0 to 1) or rich feedback.

```typescript
import { type MetricFunction } from "optiglot";

const metric: MetricFunction = (example, prediction) => {
  const isCorrect = example.label === prediction.output;
  return {
    score: isCorrect ? 1.0 : 0.0,
    feedback: isCorrect
      ? "Correctly labeled."
      : `Expected ${example.label} but got ${prediction.output}`,
  };
};
```

### 3. Run Optimization

Use `Teleprompt` to optimize the instructions in your module.

```typescript
import { Teleprompt, Dataset } from "optiglot";

// Load training data
const trainset = new Dataset(
  [
    { ticket: "I can't log into my account.", label: "Technical" },
    { ticket: "Where is my order #123?", label: "Billing" },
  ],
  ["ticket"] // Fields to pass to the forward function
);

const teleprompter = new Teleprompt("bootstrap");

// Compile: This spawns the optimization engine and finds the best prompts
const optimizedModule = await teleprompter.compile(
  classifier,
  metric,
  trainset
);

console.log("Optimized Instructions:", optimizedModule.judge.instructions);

// Save the optimized state
optimizedModule.save("./optimized_prompts.json");
```

---

## Appendix

### Persistence

You can load optimized prompts back into your module for production use:

```typescript
const productionModule = new Module({
  judge: new Predict(""), // Initial instructions don't matter if we're loading
});
productionModule.load("./optimized_prompts.json");
```

### Development

If you are developing `optiglot` locally, you can use the `OPTIGLOT_DEV=true` environment variable to run the engine from source.

```bash
OPTIGLOT_DEV=true bun run your_optimization_script.ts
```
