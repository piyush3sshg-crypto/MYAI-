"""Chart generation — hand-written SVG, no matplotlib/numpy.

matplotlib needs a C toolchain that's often painful to install on
Termux/Android. SVG is plain text, opens in any browser, and needs
nothing beyond the standard library — so this stays consistent with
the rest of the project's zero-dependency approach.
"""


def _svg_header(width, height):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
    )


def bar_chart_svg(labels, values, title="", width=500, height=320,
                   bar_color="#3378DD", value_format="{:.2f}"):
    """Vertical bar chart — for class balance, per-fold accuracy, etc."""
    if not labels:
        return _svg_header(width, height) + "</svg>"

    margin_left, margin_bottom, margin_top = 50, 60, 40
    plot_width = width - margin_left - 20
    plot_height = height - margin_bottom - margin_top
    max_value = max(values) if values else 1
    max_value = max_value if max_value > 0 else 1

    bar_width = plot_width / len(labels) * 0.7
    gap = plot_width / len(labels)

    parts = [_svg_header(width, height)]
    parts.append(
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="#1a1a1a">{title}</text>'
    )
    # axis line
    parts.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - 10}" y2="{height - margin_bottom}" stroke="#888" stroke-width="1"/>'
    )

    for i, (label, value) in enumerate(zip(labels, values)):
        bar_height = (value / max_value) * plot_height
        x = margin_left + i * gap + (gap - bar_width) / 2
        y = height - margin_bottom - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{bar_color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{x + bar_width/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#1a1a1a">{value_format.format(value)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width/2:.1f}" y="{height - margin_bottom + 18}" '
            f'text-anchor="middle" font-size="11" fill="#444">{label}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def heatmap_svg(matrix, row_labels, col_labels, title="", cell_size=60,
                 color_scale=("#EEF3FB", "#0C447C")):
    """Heatmap for a confusion matrix: matrix[row_label][col_label] = count."""
    n_rows, n_cols = len(row_labels), len(col_labels)
    margin_left, margin_top = 90, 60
    width = margin_left + n_cols * cell_size + 20
    height = margin_top + n_rows * cell_size + 20

    all_values = [matrix[r][c] for r in row_labels for c in col_labels]
    max_val = max(all_values) if all_values else 1
    max_val = max_val if max_val > 0 else 1

    def lerp_color(t):
        c0 = tuple(int(color_scale[0][i:i+2], 16) for i in (1, 3, 5))
        c1 = tuple(int(color_scale[1][i:i+2], 16) for i in (1, 3, 5))
        rgb = tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    parts = [_svg_header(width, height)]
    parts.append(
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="#1a1a1a">{title}</text>'
    )

    for ri, row in enumerate(row_labels):
        y = margin_top + ri * cell_size
        parts.append(
            f'<text x="{margin_left - 8}" y="{y + cell_size/2 + 4}" text-anchor="end" '
            f'font-size="11" fill="#444">{row}</text>'
        )
        for ci, col in enumerate(col_labels):
            x = margin_left + ci * cell_size
            value = matrix[row][col]
            t = value / max_val
            fill = lerp_color(t)
            text_color = "#ffffff" if t > 0.5 else "#1a1a1a"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" '
                f'fill="{fill}" stroke="#fff" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x + cell_size/2 - 1}" y="{y + cell_size/2 + 4}" '
                f'text-anchor="middle" font-size="13" fill="{text_color}">{value}</text>'
            )

    for ci, col in enumerate(col_labels):
        x = margin_left + ci * cell_size
        parts.append(
            f'<text x="{x + cell_size/2 - 1}" y="{margin_top - 10}" text-anchor="middle" '
            f'font-size="11" fill="#444">{col}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def line_chart_svg(values, title="", width=500, height=300, line_color="#0F6E56"):
    """Simple line chart — for loss curves over epochs."""
    if not values:
        return _svg_header(width, height) + "</svg>"

    margin_left, margin_bottom, margin_top, margin_right = 50, 40, 40, 20
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_bottom - margin_top
    max_v, min_v = max(values), min(values)
    span = (max_v - min_v) or 1

    points = []
    for i, v in enumerate(values):
        x = margin_left + (i / max(1, len(values) - 1)) * plot_width
        y = margin_top + (1 - (v - min_v) / span) * plot_height
        points.append(f"{x:.1f},{y:.1f}")

    parts = [_svg_header(width, height)]
    parts.append(
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="#1a1a1a">{title}</text>'
    )
    parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="{line_color}" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#888"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def save_svg(svg_string, path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg_string)
    return path
