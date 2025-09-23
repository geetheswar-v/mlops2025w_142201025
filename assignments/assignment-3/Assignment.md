# Assignment 3

## Question 1: Fixing JSON

```json
{
    "students": [
        {
            "id": 101,
            "name": "Sarah Johnson",
            "courses": ["CS101", "MATH200", "ENG150"],
            "gpa": 3.85,
            "active": true,
            "graduation_date": null
        },
        {
            "id": 102,
            "name": "Alex Chen",
            "courses": ["CS101", "CS102", "STAT101"],
            "gpa": 3.92,
            "active": true,
            "advisor": null,
            "notes": "Excellent student with strong analytical skills"
        },
        {
            "id": 103,
            "name": "Maria Rodriguez",
            "courses": [],
            "gpa": 3.67,
            "active": false,
            "special_programs": ["honors", "research"]
        }
    ],
    "last_updated": "2024-09-15T10:30:00Z",
    "total_students": 3
}
```

- Here i also changed string to int for id

> Even though json supports multi type, but for data integrity it is best to have in same types (for example in APIs, etc)

## Question 2

### a.How many feature flags are currently defined, and which ones are active?

- **Number defined:** 2 (`new_ui`, `analytics`).
- **Active flags:** `new_ui` (enabled = true). `analytics` is disabled (enabled = false).

### b.What happens when the log file reaches 100MB?

- When the log file reaches 100MB the logging system will rotate the file (archive/rename the current log and start a new one) because `rotate = true`.
- The concrete archive naming, compression, and retention policy depend on the logger implementation used by the application.
- Some logging might also flush the file (i don't know which lib, but normally i done like this). but logging libraries don't do this. even in production simply flusing the log(removing the old logs is not a good idea)

### c.If you wanted to make the server accessible only from localhost, what should you change?

- `"0.0.0.0"` enables all systems to use this server on any system that connect to same route
- `127.0.0.1` reserved address for localhost, doesn't leave the system (127.0.0.0/8)
- Change the `host` from `"0.0.0.0"` to `"127.0.0.1"` (or `"localhost"`). Example:

```toml
[server]
host = "127.0.0.1"
port = 8080
debug = false
max_connections = 1000
```

### d.Calculate the total number of seconds that cached items will remain valid.

- `ttl = 3600` means cached items remain valid for 3600 seconds (1 hour).

### e.Explain the difference between the [feature_flags] and [[feature_flags]] syntax

- **`[feature_flags]`**: defines a single table named `feature_flags`.
- **`[[feature_flags]]`**: defines an array of tables; each `[[...]]` block creates one element in that array (useful for listing multiple feature flag entries with their own fields).

## Question 3
[Github Link to whole](https://github.com/geetheswar-v/mlops2025w_142201025)

### instructions

```bash
git clone "https://github.com/geetheswar-v/mlops2025w_142201025"
cd mlops2025w_142201025/assignments/assignments/assignment-3
uv sync
uv run download --cifar10 # for downloading sample cifar-10 dataset
uv run inference # for 3a
uv run train # for 3d
uv run grid-search # for 3e
```

[Github Link Tree to assignment3](https://github.com/geetheswar-v/mlops2025w_142201025/tree/main/assignments/assignment-3)