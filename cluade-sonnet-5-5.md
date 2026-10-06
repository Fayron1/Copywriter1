# Claude Sonnet 5.5

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /claude/v1/messages:
    post:
      summary: Claude Sonnet 5.5
      deprecated: false
      description: >-
        ### Streaming Support


        When `stream: true` is set in the request, the API returns responses as
        server-sent events (SSE). Claude tool calling responses stream
        `tool_use` blocks and `input_json_delta` fragments.


        **Streaming Response Format:**

        - Content-Type: `text/event-stream`

        - Event names include `message_start`, `content_block_start`,
        `content_block_delta`, `message_delta`, and `message_stop`

        - Tool calls are emitted as `tool_use` content blocks

        - Final stop reason is often `tool_use` for function-calling requests


        ## Features


        - Standard chat with `messages`.

        - Function calling with `tools` and `input_schema`.

        - Optional stream response with Claude events.

        - Optional project-specific thinking flag.


        ## Request Notes


        - Put the current model name in the `model` field.

        - Use `messages` for conversation history.

        - Use `tools` to declare callable functions.

        - Set `stream` to `true` for SSE output.


        ## Authentication


        Use the auth configuration for `X-Api-Key` and `anthropic-version`. Do
        not add them as regular request parameters.
      operationId: claude_sonnet_5_5
      tags:
        - docs/en/Market/Chat  Models/Claude
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                model:
                  type: string
                  description: Model name. It must match the current document.
                  enum:
                    - claude-sonnet-5-5
                  x-apidog-enum:
                    - value: claude-sonnet-5-5
                      name: ''
                      description: ''
                  examples:
                    - claude-sonnet-5-5
                messages:
                  type: array
                  description: Conversation messages in chronological order.
                  items:
                    type: object
                    properties:
                      role:
                        type: string
                        enum:
                          - user
                          - assistant
                        description: Message role.
                        examples:
                          - user
                      content:
                        oneOf:
                          - type: string
                            description: Plain text content.
                          - type: array
                            description: Structured content blocks.
                            items:
                              type: object
                              additionalProperties: true
                              x-apidog-orders: []
                        description: Message content.
                        examples:
                          - What is the weather like in Boston today?
                    required:
                      - role
                      - content
                    x-apidog-orders:
                      - role
                      - content
                  minItems: 1
                tools:
                  type: array
                  description: >-
                    Optional callable tools. Each tool includes a name,
                    description, and input_schema.
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                        description: Function name.
                        examples:
                          - get_current_weather
                      description:
                        type: string
                        description: Human-readable function description.
                        examples:
                          - Get the current weather in a given location
                      input_schema:
                        type: object
                        description: JSON Schema for function parameters.
                        properties:
                          type:
                            type: string
                            description: Schema type.
                            examples:
                              - object
                          properties:
                            type: object
                            description: Function parameter definitions.
                            additionalProperties: true
                            x-apidog-orders: []
                            properties: {}
                          required:
                            type: array
                            description: Required parameter names.
                            items:
                              type: string
                        x-apidog-orders:
                          - type
                          - properties
                          - required
                        examples:
                          - type: object
                            properties:
                              location:
                                type: string
                                description: The city and state, e.g. Boston, MA
                            required:
                              - location
                    required:
                      - name
                      - description
                      - input_schema
                    x-apidog-orders:
                      - name
                      - description
                      - input_schema
                thinkingFlag:
                  type: boolean
                  description: >-
                    Project-specific thinking flag used by the current Claude
                    adapter.
                  examples:
                    - true
                stream:
                  type: boolean
                  default: true
                  description: If set to true, the response is returned as an SSE stream.
                  examples:
                    - false
                max_tokens:
                  type: number
                  default: 4096
                  examples:
                    - 4096
                  description: >-
                    Optional Claude output token limit. Leave empty to use the
                    default of 4096.
              required:
                - model
                - messages
              x-apidog-orders:
                - model
                - messages
                - tools
                - thinkingFlag
                - stream
                - max_tokens
              examples:
                - model: claude-opus-4-6-v1messages
                  messages:
                    - role: user
                      content: What is the weather like in Boston today?
                  tools:
                    - name: get_current_weather
                      description: Get the current weather in a given location
                      input_schema:
                        type: object
                        properties:
                          location:
                            type: string
                            description: The city and state, e.g. Boston, MA
                        required:
                          - location
                  thinkingFlag: true
                  stream: false
            example:
              model: claude-sonnet-5-5
              messages:
                - role: user
                  content: What is the weather like in Boston today?
              tools:
                - name: get_current_weather
                  description: Get the current weather in a given location
                  input_schema:
                    type: object
                    properties:
                      location:
                        type: string
                        description: The city and state, e.g. Boston, MA
                    required:
                      - location
              thinkingFlag: true
              stream: false
              max_tokens: 4096
      responses:
        '200':
          description: Request successful.
          content:
            application/json:
              schema:
                type: object
                properties:
                  role:
                    type: string
                    description: Returned message role
                    examples:
                      - assistant
                  usage:
                    type: object
                    description: Usage information returned by the provider
                    properties:
                      input_tokens:
                        type: integer
                        description: Input token count
                        examples:
                          - 600
                      output_tokens:
                        type: integer
                        description: Output token count
                        examples:
                          - 57
                      cache_creation_input_tokens:
                        type: integer
                        description: Cache creation input token count
                        examples:
                          - 0
                      cache_read_input_tokens:
                        type: integer
                        description: Cache read input token count
                        examples:
                          - 0
                      service_tier:
                        type: string
                        description: Service tier
                        examples:
                          - standard
                    x-apidog-orders:
                      - input_tokens
                      - output_tokens
                      - cache_creation_input_tokens
                      - cache_read_input_tokens
                      - service_tier
                  stop_reason:
                    type: string
                    description: Reason why generation stopped
                    examples:
                      - tool_use
                  model:
                    type: string
                    description: Actual model version returned by the provider
                    examples:
                      - claude-opus-4-5-20251101
                  id:
                    type: string
                    description: Unique message identifier
                    examples:
                      - msg_01VSoxV4a8YWB3DBh9TdM63W
                  credits_consumed:
                    type: number
                    description: Credits consumed by the request
                    examples:
                      - 0.25
                  type:
                    type: string
                    description: Top-level response object type
                    examples:
                      - message
                  content:
                    type: array
                    description: Response content blocks
                    items:
                      type: object
                      properties:
                        input:
                          type: object
                          description: Tool input arguments
                          additionalProperties: true
                          x-apidog-orders: []
                          properties: {}
                        caller:
                          type: object
                          description: Tool caller metadata
                          properties:
                            type:
                              type: string
                              examples:
                                - direct
                          x-apidog-orders:
                            - type
                        name:
                          type: string
                          description: Tool name
                          examples:
                            - get_current_weather
                        id:
                          type: string
                          description: Tool call identifier
                          examples:
                            - toolu_018gdqs2FHxrRjQHLZv1qvbF
                        type:
                          type: string
                          description: Content block type
                          examples:
                            - tool_use
                      x-apidog-orders:
                        - input
                        - caller
                        - name
                        - id
                        - type
                x-apidog-orders:
                  - role
                  - usage
                  - stop_reason
                  - model
                  - id
                  - credits_consumed
                  - type
                  - content
              example:
                role: assistant
                usage:
                  cache_creation:
                    ephemeral_1h_input_tokens: 0
                    ephemeral_5m_input_tokens: 0
                  output_tokens: 57
                  service_tier: standard
                  cache_creation_input_tokens: 0
                  input_tokens: 600
                  cache_read_input_tokens: 0
                  inference_geo: not_available
                stop_reason: tool_use
                model: claude-sonnet-5-5
                id: msg_01VSoxV4a8YWB3DBh9TdM63W
                credits_consumed: 0.25
                type: message
                content:
                  - input:
                      location: Beijing, China
                    caller:
                      type: direct
                    name: get_current_weather
                    id: toolu_018gdqs2FHxrRjQHLZv1qvbF
                    type: tool_use
          headers: {}
          x-apidog-name: ''
        '400':
          description: Bad Request - Invalid request parameters
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: object
                    properties:
                      message:
                        type: string
                        examples:
                          - Invalid request parameters
                      type:
                        type: string
                        examples:
                          - invalid_request_error
                    x-apidog-orders:
                      - message
                      - type
                x-apidog-orders:
                  - error
          headers: {}
          x-apidog-name: ''
        '401':
          description: Unauthorized - Invalid or missing API key
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: object
                    properties:
                      message:
                        type: string
                        examples:
                          - Invalid or missing API key
                      type:
                        type: string
                        examples:
                          - authentication_error
                    x-apidog-orders:
                      - message
                      - type
                x-apidog-orders:
                  - error
          headers: {}
          x-apidog-name: ''
      security: []
      x-apidog-folder: docs/en/Market/Chat  Models/Claude
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-44511365-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```
