# uv run python src/try_something/demo_arg_parse.py --name zzy --repeat-count 5 --style long
import argparse

parser = argparse.ArgumentParser(description="参数解析示例")

parser.add_argument("--name", type=str, required=True)
parser.add_argument("--repeat-count", type=int, default=3)
parser.add_argument(
    "--style",
    type=str,
    choices=["short", "long"],
    default="short",
)

args = parser.parse_args()

print(args.name)
print(args.repeat_count)
print(args.style)