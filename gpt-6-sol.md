# GPT 6 Sol

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /codex/v1/responses:
    post:
      summary: GPT 6 Sol
      deprecated: false
      description: >-
        > GPT 6 Sol API is a multimodal chat-completions style endpoint that
        accepts structured input arrays, supports adjustable reasoning effort,
        and integrates web search or function calling tools.



        <CardGroup cols={2}>
          <Card title="Multimodal Input" icon="🖼️">
            Supports mixed text, image, and file inputs in a single message.
          </Card>

          <Card title="Reasoning Control" icon="🧠">
            Adjustable reasoning effort from low to xhigh.
          </Card>

          <Card title="Tools & Web Search" icon="✨">
            Integrates web search or custom function calling tools.
          </Card>

          <Card title="Unified Endpoint" icon="💻">
            Uses the unified <code>/codex/v1/responses</code> endpoint with <code>model</code> set to one of <code>gpt-6-sol</code>.
          </Card>
        </CardGroup>


        ## Tools & tool\_choice


        The `tools` array enables **web search** or **function calling**
        capabilities.


        :::caution

        Web Search and Function Calling are **mutually exclusive**.  

        In a single request you should choose only one: do not include both
        `{"type": "web_search"}` and `{"type": "function", ...}` in the same
        `tools` array.

        :::



        <AccordionGroup>
          <Accordion title="Web Search">
            Use the built-in Web Search tool to retrieve up-to-date information:

            ```json
            {
              "tools": [
                {
                  "type": "web_search"
                }
              ]
            }
            ```
          </Accordion>

          <Accordion title="Function Calling">
            Define business functions that the model can call when needed:

            ```json
            {
              "tools": [
                {
                  "type": "function",
                  "name": "get_current_weather",
                  "description": "Get the current weather in a given location",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                      },
                      "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                      }
                    },
                    "required": ["location", "unit"]
                  }
                }
              ],
              "tool_choice": "auto"
            }
            ```

            When function tools are configured in `tools`, set `tool_choice` to `"auto"` so the model can decide when to call them.  
            If you do not configure any function tools, omit the `tool_choice` field.
          </Accordion>
        </AccordionGroup>
      operationId: gpt-6-sol-completions
      tags:
        - docs/en/Market/Chat  Models/GPT
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                model:
                  type: string
                  description: 'Target model name. Allowed values: `gpt-6-sol`.'
                  default: gpt-6-sol
                  enum:
                    - gpt-6-sol
                  x-apidog-enum:
                    - value: gpt-6-sol
                      name: ''
                      description: ''
                  examples:
                    - gpt-6-sol
                stream:
                  type: boolean
                  description: >-
                    When true, responses stream in real time as server-sent
                    events. When false, the full response is returned at once
                    after completion. Default is true.
                  default: false
                input:
                  oneOf:
                    - type: string
                      description: Input can be a string.
                    - type: array
                      description: >-
                        Input can be an array; each element is a message object
                        with role and content.
                      items:
                        type: object
                        properties: {}
                      minItems: 1
                reasoning:
                  type: object
                  description: Reasoning configuration for the model.
                  properties:
                    effort:
                      type: string
                      description: >-
                        Reasoning effort level. Higher values provide more
                        thorough reasoning but may increase latency. Defaults to
                        "low".
                      enum:
                        - low
                        - medium
                        - high
                        - xhigh
                      default: low
                      examples:
                        - low
                      x-apidog-enum:
                        - value: low
                          name: ''
                          description: ''
                        - value: medium
                          name: ''
                          description: ''
                        - value: high
                          name: ''
                          description: ''
                        - value: xhigh
                          name: ''
                          description: ''
                  x-apidog-orders:
                    - effort
                  x-apidog-ignore-properties: []
                tools:
                  type: array
                  description: >-
                    Optional tools that the model may call. Either web search OR
                    function calling should be configured, but not both
                    simultaneously.
                  items:
                    oneOf:
                      - $ref: '#/components/schemas/ToolWebSearch'
                      - $ref: '#/components/schemas/ToolFunction'
                tool_choice:
                  type: string
                  description: >-
                    Tool selection behavior. When function tools are configured
                    in `tools`, set this to `auto` so the model can decide when
                    to call them.
                  examples:
                    - auto
              required:
                - model
                - input
              x-apidog-orders:
                - model
                - stream
                - input
                - reasoning
                - tools
                - tool_choice
              examples:
                - model: gpt-5.1-codex
                  input:
                    - role: user
                      content:
                        - type: input_text
                          text: What is in this image?
                        - type: input_image
                          image_url: >-
                            https://file.aiquickdraw.com/custom-page/akr/section-images/1759055072437dqlsclj2.png
                  tools:
                    - type: web_search
                  reasoning:
                    effort: high
              x-apidog-ignore-properties: []
            example:
              model: gpt-6-sol
              input:
                - role: user
                  content:
                    - type: input_text
                      text: What is in this image?
                    - type: input_image
                      image_url: >-
                        https://file.aiquickdraw.com/custom-page/akr/section-images/1759055072437dqlsclj2.png
              tools:
                - type: web_search
              reasoning:
                effort: high
      responses:
        '200':
          description: 'Request successful. '
          content:
            text/event-stream:
              schema:
                type: string
                description: >-
                  Streaming responses are sent as Server-Sent Events (SSE) with
                  `Content-Type: text/event-stream`.


                  **Standard output**


                  - **Text delta event**: `event: response.output_text.delta`
                    - `data.delta`: The incremental text content in the stream
                    - `data.type`: Event type, always `response.output_text.delta`
                  - **Completion event**: `event: response.completed`
                    - `data.response.usage`: Token usage information, such as `input_tokens` and `output_tokens`

                  **Function Calling**


                  - **Function call arguments delta event**: `event:
                  response.function_call_arguments.delta`
                    - `data.delta`: Incremental string content of the function call arguments
                    - `data.type`: Event type, always `response.function_call_arguments.delta`
                  - **Completion event**: `event: response.completed`
                    - `data.response.usage`: Token usage information, such as `input_tokens` and `output_tokens`

                  The final line `data: [DONE]` is the stream end marker,
                  indicating that no more events will be sent.
              example: |-
                {
                  "output": [
                    {
                      "type": "reasoning",
                      "id": "rs_xxx",
                      "summary": []
                    },
                    {
                      "type": "message",
                      "role": "assistant",
                      "id": "msg_xxx",
                      "content": [
                        {
                          "type": "output_text",
                          "text": "Hello! How can I help you today?"
                        }
                      ],
                      "status": "completed"
                    }
                  ],
                  "usage": {
                    "total_tokens": 4490,
                    "output_tokens": 47,
                    "input_tokens": 4443
                  },
                  "credits_consumed": 0.48,
                  "status": "completed"
                }
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
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - error
                x-apidog-ignore-properties: []
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
                          - Unauthorized
                      type:
                        type: string
                        examples:
                          - authentication_error
                    x-apidog-orders:
                      - message
                      - type
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - error
                x-apidog-ignore-properties: []
          headers: {}
          x-apidog-name: ''
        '429':
          description: Rate Limited - Too many requests
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
                          - Rate limit exceeded
                      type:
                        type: string
                        examples:
                          - rate_limit_error
                    x-apidog-orders:
                      - message
                      - type
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - error
                x-apidog-ignore-properties: []
          headers: {}
          x-apidog-name: ''
        '500':
          description: 请求失败
          content:
            application/json:
              schema:
                type: object
                properties: {}
                x-apidog-orders: []
                x-apidog-ignore-properties: []
          headers: {}
          x-apidog-name: Error
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
      x-apidog-folder: docs/en/Market/Chat  Models/GPT
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-44151087-run
components:
  schemas:
    ToolFunction:
      type: object
      description: Function call tool definition.
      properties:
        type:
          type: string
          enum:
            - function
          examples:
            - function
        name:
          type: string
          description: Function name.
          examples:
            - get_current_weather
        description:
          type: string
          description: A readable description of the function's purpose.
        parameters:
          type: object
          description: JSON Schema describing function parameters.
          x-apidog-orders: []
          properties: {}
          x-apidog-ignore-properties: []
      required:
        - type
        - name
        - description
        - parameters
      x-apidog-orders:
        - type
        - name
        - description
        - parameters
      x-apidog-ignore-properties: []
      x-apidog-folder: ''
    ToolWebSearch:
      type: object
      description: Online search tool configuration.
      properties:
        type:
          type: string
          enum:
            - web_search
          examples:
            - web_search
      required:
        - type
      x-apidog-orders:
        - type
      x-apidog-ignore-properties: []
      x-apidog-folder: ''
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
