from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def build_report(result_data, templates_dir, output_dir):
    """Render an HTML report from the analysis results."""
    environment = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.get_template("report_template.html")

    report_html = template.render(result=result_data)
    report_path = Path(output_dir) / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path
