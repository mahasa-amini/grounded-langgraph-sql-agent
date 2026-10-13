from agent.graph import build_graph


app = build_graph()


def print_mermaid_graph() -> None:
    print("\n=== MERMAID GRAPH ===\n")
    print(app.get_graph().draw_mermaid())


def print_ascii_graph() -> None:
    print("\n=== ASCII GRAPH ===\n")
    app.get_graph().print_ascii()


if __name__ == "__main__":
    print_mermaid_graph()
    print_ascii_graph()