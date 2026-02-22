---
name: pydantic-ai
description: "Build production-grade AI agents with Pydantic AI framework. Use when users want to: create AI agents with multiple model providers (OpenAI, Anthropic, Google/Gemini, Cohere, etc.), implement function calling/tools with automatic schema generation, build RAG (Retrieval-Augmented Generation) systems, switch between models easily for testing/comparison, create type-safe structured outputs, implement dependency injection for databases/APIs, build graph-based agent workflows, or migrate from LangChain/LlamaIndex to a more Pythonic framework. Also use for questions about Pydantic AI syntax, best practices, or troubleshooting."
license: Proprietary
---

# Pydantic AI - Agent Framework

Build production-grade AI agents with type safety, multi-model support, and powerful function calling.

## Overview

Pydantic AI is a Python agent framework designed for production applications. Built by the Pydantic team, it provides:

- **Model-agnostic design** - Single interface for 20+ providers
- **Type safety first** - Pydantic models for inputs/outputs
- **Powerful function calling** - Automatic schema generation from Python functions
- **Production-ready** - Built-in retry logic, streaming, observability

## Quick Reference

| Task                  | Code Pattern                                                     |
| --------------------- | ---------------------------------------------------------------- |
| **Basic agent**       | `Agent('google:gemini-3-flash-preview')`                         |
| **Run agent**         | `result = agent.run_sync('query')` or `await agent.run('query')` |
| **Add tool**          | `@agent.tool` decorator on async function                        |
| **Structured output** | `Agent(model, output_type=MyPydanticModel)`                      |
| **Dependencies**      | `Agent(model, deps_type=MyDeps)` + `deps=my_deps`                |
| **Switch models**     | Change model string or `model='other-model'` at runtime          |

### Model Strings

```python
# Format: 'provider:model-name'
'google:gemini-3-flash-preview'           # Gemini via API
'google-vertex:gemini-3-flash-preview'    # Gemini via Vertex AI
'cohere:command-r-plus-08-2024'           # Cohere
'openai:gpt-4o'                           # OpenAI
'anthropic:claude-sonnet-4-5'             # Anthropic
'gateway/openai:gpt-4o'                   # Via Pydantic Gateway
```

### Common Imports

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from dataclasses import dataclass
```

---

## Installation

```bash
# Full installation
pip install pydantic-ai

# Slim with specific providers (recommended)
pip install pydantic-ai-slim[openai,google,cohere]

# For your project
pip install pydantic-ai-slim[google,cohere]  # Gemini + Cohere only
```

**Optional dependencies:**

```bash
# For RAG/database work
pip install asyncpg pgvector  # PostgreSQL + vector search

# For graph workflows
pip install pydantic-graph

# For observability
pip install logfire
```

**Environment variables:**

```bash
export GEMINI_API_KEY='your-key'
export COHERE_API_KEY='your-key'
export OPENAI_API_KEY='your-key'  # If using OpenAI embeddings
```

---

## Core Patterns

### 1. Basic Agent

```python
from pydantic_ai import Agent

# Create agent
agent = Agent(
    'google:gemini-3-flash-preview',
    instructions='You are a helpful assistant.',
)

# Synchronous
result = agent.run_sync('What is 2+2?')
print(result.output)  # "4"

# Asynchronous (preferred for production)
result = await agent.run('What is 2+2?')
print(result.output)
```

### 2. Structured Output

**Use Pydantic models to guarantee output structure:**

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str

agent = Agent(
    'google:gemini-3-flash-preview',
    output_type=Answer,  # Forces this structure
)

result = agent.run_sync('What is the capital of France?')
# result.output is guaranteed to be an Answer instance
assert isinstance(result.output, Answer)
print(f"Answer: {result.output.answer}")
print(f"Confidence: {result.output.confidence}")
```

### 3. Function Calling (Tools)

**Tools let agents take actions and access data:**

```python
from pydantic_ai import Agent, RunContext

agent = Agent('google:gemini-3-flash-preview')

@agent.tool
async def search_database(
    ctx: RunContext[None],
    query: str,
) -> list[dict]:
    """
    Search the database for relevant information.

    Args:
        query: Search query

    Returns:
        List of matching results
    """
    # Your database search logic
    results = await db.search(query)
    return results

# The model can now call this tool automatically
result = await agent.run('Find all users in San Francisco')
```

**Key points:**

- Docstrings become tool descriptions (Google/NumPy/Sphinx formats supported)
- Type hints define parameter schemas
- Schema is auto-generated and sent to the model
- Return value is sent back to model for reasoning

### 4. Dependencies (Database Connections, APIs)

**Use typed dependencies for clean architecture:**

```python
from dataclasses import dataclass
import asyncpg

@dataclass
class DatabaseDeps:
    """Dependencies for database access."""
    pool: asyncpg.Pool

    async def query(self, sql: str, *args):
        async with self.pool.acquire() as conn:
            return await conn.fetch(sql, *args)

# Agent with typed dependencies
agent = Agent(
    'google:gemini-3-flash-preview',
    deps_type=DatabaseDeps,
)

@agent.tool
async def get_user(
    ctx: RunContext[DatabaseDeps],
    user_id: int,
) -> dict:
    """Get user information by ID."""
    results = await ctx.deps.query(
        'SELECT * FROM users WHERE id = $1',
        user_id
    )
    return dict(results[0]) if results else None

# Run with dependencies
pool = await asyncpg.create_pool('postgresql://...')
deps = DatabaseDeps(pool=pool)

result = await agent.run('Get user 123', deps=deps)
```

### 5. Multi-Model Support

**Switch models with a single line change:**

```python
# Method 1: Factory pattern (recommended)
def create_agent(model: str):
    return Agent(
        model,
        instructions='You are a helpful assistant.',
        # Tools and config stay the same
    )

gemini_agent = create_agent('google:gemini-3-flash-preview')
cohere_agent = create_agent('cohere:command-r-plus-08-2024')

# Method 2: Runtime override
agent = Agent('google:gemini-3-flash-preview')
result = await agent.run(
    'Your query',
    model='cohere:command-r-plus-08-2024'  # Override
)

# Method 3: Fallback on failure
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel([
    'google:gemini-3-flash-preview',
    'cohere:command-r-plus-08-2024',  # Fallback
])
agent = Agent(model)
```

---

## Implementation Patterns

### RAG (Retrieval-Augmented Generation)

**Combine retrieval with generation for accurate responses:**

```python
from openai import AsyncOpenAI

@dataclass
class RAGDeps:
    db_pool: asyncpg.Pool
    openai_client: AsyncOpenAI

agent = Agent('google:gemini-3-flash-preview', deps_type=RAGDeps)

@agent.tool
async def search_knowledge_base(
    ctx: RunContext[RAGDeps],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search knowledge base using semantic similarity.

    Args:
        query: What to search for
        top_k: Number of results

    Returns:
        Relevant documents with metadata
    """
    # 1. Generate embedding
    response = await ctx.deps.openai_client.embeddings.create(
        model='text-embedding-3-small',
        input=query
    )
    embedding = response.data[0].embedding

    # 2. Vector search (requires pgvector extension)
    async with ctx.deps.db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT id, content, metadata,
                   embedding <-> $1 AS distance
            FROM documents
            ORDER BY distance
            LIMIT $2
            """,
            embedding,
            top_k
        )

    return [dict(r) for r in results]

# Usage
result = await agent.run(
    'What is our refund policy?',
    deps=RAGDeps(db_pool=pool, openai_client=openai)
)
```

**Database setup for RAG:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(1536)  -- OpenAI embedding dimension
);

CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops);
```

### Graph-RAG Pattern

**For complex relationship-based retrieval (your use case):**

```python
from neo4j import AsyncGraphDatabase

@dataclass
class GraphRAGDeps:
    neo4j_driver: AsyncGraphDatabase
    pg_pool: asyncpg.Pool
    openai_client: AsyncOpenAI

    async def graph_query(self, cypher: str, **params):
        async with self.neo4j_driver.session() as session:
            result = await session.run(cypher, **params)
            return [record.data() async for record in result]

agent = Agent('google:gemini-3-flash-preview', deps_type=GraphRAGDeps)

@agent.tool
async def find_spatial_relations(
    ctx: RunContext[GraphRAGDeps],
    entity_type: str,
    relation: str,
    reference: str,
) -> list[dict]:
    """
    Find entities with spatial relationships.

    Args:
        entity_type: Type of entity to find
        relation: Spatial relationship (near, above, below, etc.)
        reference: Reference object

    Returns:
        Related entities with positions
    """
    cypher = f"""
    MATCH (ref {{name: $ref}})-[:{relation.upper()}]->(target:{entity_type})
    RETURN target.name AS name,
           target.x AS x, target.y AS y, target.z AS z,
           target.description AS description
    """
    return await ctx.deps.graph_query(cypher, ref=reference)

@agent.tool
async def semantic_search_entities(
    ctx: RunContext[GraphRAGDeps],
    query: str,
) -> list[dict]:
    """
    Search entities using semantic similarity.

    Args:
        query: Natural language description

    Returns:
        Matching entities
    """
    # Generate embedding
    response = await ctx.deps.openai_client.embeddings.create(
        model='text-embedding-3-small',
        input=query
    )
    embedding = response.data[0].embedding

    # Vector search
    async with ctx.deps.pg_pool.acquire() as conn:
        candidates = await conn.fetch(
            """
            SELECT name, entity_type, x, y, z, description
            FROM spatial_entities
            ORDER BY embedding <-> $1
            LIMIT 10
            """,
            embedding
        )

    return [dict(c) for c in candidates]
```

### Multiple Tools Pattern

```python
agent = Agent('google:gemini-3-flash-preview', deps_type=MyDeps)

@agent.tool
async def tool_1(ctx: RunContext[MyDeps], arg: str) -> str:
    """First tool description."""
    return await ctx.deps.do_something(arg)

@agent.tool
async def tool_2(ctx: RunContext[MyDeps], x: int, y: int) -> int:
    """Second tool description."""
    return await ctx.deps.calculate(x, y)

@agent.tool
async def tool_3(ctx: RunContext[MyDeps]) -> list[dict]:
    """Third tool description."""
    return await ctx.deps.fetch_data()

# Model automatically chooses which tools to call
result = await agent.run('Complex query', deps=deps)
```

### Streaming Responses

```python
agent = Agent('google:gemini-3-flash-preview')

async with agent.run_stream('Write a long story') as stream:
    async for chunk in stream.stream_text():
        print(chunk, end='', flush=True)

    # Get final result
    result = await stream.get_result()
    print(f"\n\nTotal tokens: {result.usage().total_tokens}")
```

---

## Model-Specific Features

### Google Gemini

**Installation:**

```bash
pip install pydantic-ai-slim[google]
```

**Basic usage:**

```python
agent = Agent('google:gemini-3-flash-preview')
```

**Advanced configuration:**

```python
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

# Via Vertex AI (production recommended)
provider = GoogleProvider(
    vertexai=True,
    project='your-gcp-project',
)
model = GoogleModel('gemini-3-flash-preview', provider=provider)

# Model settings
settings = GoogleModelSettings(
    # Thinking mode for complex reasoning
    google_thinking_config={'thinking_level': 'high'},

    # Safety settings
    google_safety_settings={
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
    },
)

agent = Agent(model, model_settings=settings)
```

**Thinking mode (for complex reasoning):**

```python
settings = GoogleModelSettings(
    google_thinking_config={'thinking_level': 'high'}
)
agent = Agent('google:gemini-3-flash-preview', model_settings=settings)

# Use for complex multi-step problems
result = await agent.run('Solve this complex spatial reasoning task...')
```

### Cohere

**Installation:**

```bash
pip install pydantic-ai-slim[cohere]
```

**Basic usage:**

```python
agent = Agent('cohere:command-r-plus-08-2024')
```

**Best for:**

- Long-context tasks (128k tokens)
- Multi-language support
- Enterprise use cases

---

## Best Practices

### 1. Use Async by Default

```python
# ✅ GOOD - Async (non-blocking)
result = await agent.run('query', deps=deps)

# ❌ AVOID - Sync (blocks thread)
result = agent.run_sync('query', deps=deps)
```

### 2. Structured Dependencies

```python
# ✅ GOOD - Type-safe dataclass
@dataclass
class Deps:
    db: Database
    cache: Redis

agent = Agent(model, deps_type=Deps)

# ❌ AVOID - Dict or loose typing
agent = Agent(model)  # No deps_type
```

### 3. Explicit Tool Descriptions

```python
# ✅ GOOD - Clear docstring with Args/Returns
@agent.tool
async def search(ctx: RunContext[Deps], query: str) -> list[dict]:
    """
    Search the database for matching records.

    Args:
        query: Search query string

    Returns:
        List of matching records
    """
    pass

# ❌ AVOID - No docstring
@agent.tool
async def search(ctx: RunContext[Deps], query: str):
    pass
```

### 4. Model Abstraction

```python
# ✅ GOOD - Configurable model
MODEL = os.getenv('AI_MODEL', 'google:gemini-3-flash-preview')
agent = Agent(MODEL)

# ❌ AVOID - Hard-coded model
agent = Agent('google:gemini-3-flash-preview')
```

### 5. Error Handling

```python
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior

@agent.tool(retries=3)
async def unreliable_api(ctx: RunContext[None], query: str):
    """Call external API with retry logic."""
    try:
        result = await external_api.call(query)
        return result
    except TransientError as e:
        # Retry on transient errors
        raise ModelRetry(f"API unavailable: {e}")
    except PermanentError as e:
        # Don't retry on permanent errors
        raise
```

### 6. Cost Tracking

```python
from pydantic_ai import UsageLimits

limits = UsageLimits(
    request_limit=100,
    total_tokens_limit=100000,
    total_cost_limit=10.0,  # Max $10
)

result = await agent.run('query', usage_limits=limits)

# Check usage
usage = result.usage()
print(f"Tokens: {usage.total_tokens}")
print(f"Cost: ${usage.total_cost}")
print(f"Requests: {usage.requests}")
```

### 7. Observability (Logfire)

```python
import logfire

# One-time setup
logfire.configure()
logfire.instrument_pydantic_ai()

# All agent runs now automatically tracked
result = await agent.run('query', deps=deps)

# View in Logfire dashboard:
# - Request/response times
# - Token usage & costs
# - Tool calls
# - Errors
```

---

## Common Patterns for Your Use Case

### Spatial Query Agent with Multi-Model Testing

```python
from pydantic import BaseModel
from typing import Literal

class SpatialCoordinate(BaseModel):
    x: float
    y: float
    z: float

class SpatialEntity(BaseModel):
    name: str
    entity_type: str
    position: SpatialCoordinate
    description: str

class SpatialQueryResult(BaseModel):
    entities: list[SpatialEntity]
    reasoning: str
    confidence: float

@dataclass
class SpatialDeps:
    neo4j_driver: AsyncGraphDatabase
    pg_pool: asyncpg.Pool

def create_spatial_agent(model: str):
    """Factory for spatial agents with different models."""
    agent = Agent(
        model,
        deps_type=SpatialDeps,
        output_type=SpatialQueryResult,
        instructions="""
        You are a spatial reasoning assistant.
        Use the tools to query the graph database.
        Always provide clear reasoning.
        """,
    )

    @agent.tool
    async def find_nearby(
        ctx: RunContext[SpatialDeps],
        reference: str,
        max_distance: float,
    ) -> list[dict]:
        """Find objects within distance of reference."""
        cypher = """
        MATCH (ref {name: $ref})-[r:NEAR]->(target)
        WHERE r.distance <= $max_dist
        RETURN target
        """
        return await ctx.deps.neo4j_query(cypher,
            ref=reference, max_dist=max_distance)

    return agent

# Test multiple models
async def compare_models(query: str, deps: SpatialDeps):
    models = [
        'google:gemini-3-flash-preview',
        'cohere:command-r-plus-08-2024',
    ]

    for model_name in models:
        agent = create_spatial_agent(model_name)
        result = await agent.run(query, deps=deps)

        print(f"\n{model_name}:")
        print(f"Entities: {len(result.output.entities)}")
        print(f"Confidence: {result.output.confidence}")
        print(f"Cost: ${result.usage().total_cost}")
```

---

## Troubleshooting

### Tool Not Being Called

**Problem:** Model doesn't call your tool.

**Solutions:**

1. Add clear docstring with Args/Returns sections
2. Make parameter names descriptive
3. Use type hints (required)
4. Test with `@agent.tool_plain` first (no context)
5. Check model supports function calling (all major models do)

### Dependency Injection Not Working

**Problem:** `ctx.deps` is None or wrong type.

**Solutions:**

1. Set `deps_type` on Agent: `Agent(model, deps_type=MyDeps)`
2. Pass `deps=` when calling: `agent.run('query', deps=my_deps)`
3. Use dataclass (not dict) for dependencies

### Model Hallucinating Tool Results

**Problem:** Model invents data instead of calling tools.

**Solutions:**

1. Make instructions explicit: "ALWAYS use the search_database tool"
2. Use structured output to force tool usage
3. Try different model (some are better at tool use)
4. Validate output and retry with `ModelRetry` if invalid

### Cost Too High

**Solutions:**

1. Use cheaper models: Gemini Flash < Gemini Pro < GPT-4
2. Set `UsageLimits` to cap costs
3. Cache expensive operations
4. Use streaming to stop early if needed
5. Limit context size (shorter system prompts)

### Rate Limits

**Solutions:**

1. Use `ConcurrencyLimitedModel` to limit parallel requests
2. Add retry logic with exponential backoff
3. Switch to different provider (Vertex AI has higher limits)

---

## Testing

### Unit Test Individual Tools

```python
import pytest
from pydantic_ai import RunContext

@pytest.mark.asyncio
async def test_search_tool():
    # Create mock context
    ctx = RunContext(deps=None, messages=[], model_info=None)

    # Test tool directly
    result = await search_database(ctx, 'test query')

    assert isinstance(result, list)
    assert len(result) > 0
```

### Mock LLM for Testing

```python
from pydantic_ai.models.test import TestModel

def test_agent_workflow():
    # Mock model with predetermined response
    test_model = TestModel()
    test_model.custom_result_text = "Expected output"

    agent = Agent(test_model)
    result = agent.run_sync('test input')

    assert result.output == "Expected output"
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_full_workflow():
    # Use real model but with test data
    agent = create_spatial_agent('google:gemini-3-flash-preview')

    result = await agent.run(
        'Find objects near test_object',
        deps=test_deps
    )

    assert isinstance(result.output, SpatialQueryResult)
    assert result.output.confidence > 0.5
```

---

## Critical Rules

1. **Always use async** - `await agent.run()` not `agent.run_sync()`
2. **Set deps_type** - Enable type checking: `Agent(model, deps_type=MyDeps)`
3. **Document tools** - Docstrings become schemas (Google/NumPy/Sphinx)
4. **Type everything** - Type hints are required for parameters
5. **Handle errors** - Use `ModelRetry` for transient failures
6. **Track costs** - Use `UsageLimits` and check `result.usage()`
7. **Test thoroughly** - Unit test tools, integration test full workflows
8. **Use structured outputs** - `output_type=MyModel` for consistency
9. **Abstract models** - Don't hard-code provider, use config/env vars
10. **Monitor in production** - Use Logfire or similar for observability

---

## Quick Checklist

Before deploying to production:

- [ ] Using async (`await agent.run()`)
- [ ] Dependencies are dataclasses with type hints
- [ ] All tools have docstrings with Args/Returns
- [ ] Structured output types defined (if applicable)
- [ ] Error handling with retries on transient failures
- [ ] Usage limits configured
- [ ] Model selection is configurable (env var or config)
- [ ] Logging/observability configured
- [ ] Tests written (unit + integration)
- [ ] Cost tracking enabled

---

## Example Project Structure

```
my_spatial_agent/
├── agents/
│   ├── __init__.py
│   └── spatial_agent.py       # Agent factory
├── models/
│   ├── __init__.py
│   └── schemas.py              # Pydantic models
├── tools/
│   ├── __init__.py
│   ├── graph_search.py         # Graph DB tools
│   └── vector_search.py        # Vector search tools
├── deps.py                     # Dependency classes
├── config.py                   # Configuration
├── tests/
│   ├── test_tools.py
│   └── test_agent.py
└── main.py                     # Entry point
```

**Example `deps.py`:**

```python
from dataclasses import dataclass
import asyncpg
from neo4j import AsyncGraphDatabase

@dataclass
class AppDeps:
    neo4j: AsyncGraphDatabase
    postgres: asyncpg.Pool

    async def close(self):
        await self.neo4j.close()
        await self.postgres.close()
```

**Example `config.py`:**

```python
import os

MODEL = os.getenv('AI_MODEL', 'google:gemini-3-flash-preview')
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
POSTGRES_URI = os.getenv('POSTGRES_URI', 'postgresql://...')
```

---

## Additional Resources

- **Official Docs:** https://ai.pydantic.dev
- **Models Guide:** https://ai.pydantic.dev/models/overview
- **Tools Guide:** https://ai.pydantic.dev/tools
- **RAG Example:** https://ai.pydantic.dev/examples/rag
- **GitHub:** https://github.com/pydantic/pydantic-ai
