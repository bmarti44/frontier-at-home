/* Test-only W7 oracle. The build must put the frozen engine source directory
 * on the include path. Renamed narrow stubs let this translation unit execute
 * the real static completion parser and GLM renderer without loading a model.
 */
#define DS4_SERVER_TEST 1
#define DS4_SERVER_TEST_NO_MAIN 1
#define ds4_engine_is_glm_dsa w7_engine_is_glm_dsa
#define ds4_think_mode_enabled w7_think_mode_enabled
#define ds4_glm_reasoning_effort_text w7_glm_reasoning_effort_text
#define ds4_think_mode_for_context w7_think_mode_for_context
#define ds4_tokenize_rendered_chat w7_tokenize_rendered_chat
#define ds4_tokens_free w7_tokens_free
#include "ds4_server.c"

bool w7_engine_is_glm_dsa(ds4_engine *engine) {
    return engine != NULL;
}

bool w7_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char *w7_glm_reasoning_effort_text(ds4_think_mode mode) {
    if (mode == DS4_THINK_HIGH) return "Reasoning Effort: High";
    if (mode == DS4_THINK_MAX) return "Reasoning Effort: Max";
    return NULL;
}

ds4_think_mode w7_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && ctx_size < 131072) return DS4_THINK_HIGH;
    return mode;
}

void w7_tokenize_rendered_chat(ds4_engine *engine, const char *text,
                               ds4_tokens *out) {
    (void)engine;
    (void)text;
    if (out) memset(out, 0, sizeof(*out));
}

void w7_tokens_free(ds4_tokens *tokens) {
    if (tokens) memset(tokens, 0, sizeof(*tokens));
}

static char *read_stdin_all(void) {
    size_t length = 0;
    size_t capacity = 16384;
    char *body = malloc(capacity + 1);
    if (!body) return NULL;
    for (;;) {
        if (length == capacity) {
            if (capacity > SIZE_MAX / 2) {
                free(body);
                return NULL;
            }
            capacity *= 2;
            char *grown = realloc(body, capacity + 1);
            if (!grown) {
                free(body);
                return NULL;
            }
            body = grown;
        }
        size_t count = fread(body + length, 1, capacity - length, stdin);
        length += count;
        if (count == 0) {
            if (ferror(stdin)) {
                free(body);
                return NULL;
            }
            break;
        }
    }
    body[length] = '\0';
    return body;
}

int main(void) {
    char error[256] = {0};
    char *body = read_stdin_all();
    if (!body) {
        fputs("failed to read request body\n", stderr);
        return 2;
    }
    request parsed;
    if (!parse_completion_request((ds4_engine *)(uintptr_t)1, body, 0, 8192,
                                  &parsed, error, sizeof(error))) {
        fprintf(stderr, "completion parse failed: %s\n", error);
        free(body);
        return 2;
    }
    const size_t bytes = strlen(parsed.prompt_text);
    if (fwrite(parsed.prompt_text, 1, bytes, stdout) != bytes) {
        free(parsed.prompt_text);
        free(parsed.model);
        free(body);
        return 2;
    }
    free(parsed.prompt_text);
    free(parsed.model);
    free(body);
    return 0;
}
