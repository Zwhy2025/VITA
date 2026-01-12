import re
import os
import sys

def analyze_diff(diff_file):
    if not os.path.exists(diff_file):
        print(f"Error: File {diff_file} not found.")
        return

    with open(diff_file, 'r', encoding='utf-8') as f:
        content = f.read()

    files = {}
    current_file = None
    
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git'):
            match = re.search(r'b/(.*)', line)
            if match:
                current_file = match.group(1).split(' ')[0] # 处理可能存在的空格
                files[current_file] = {'status': 'modified', 'key_changes': []}
            i += 1
        elif line.startswith('new file mode'):
            if current_file:
                files[current_file]['status'] = 'new'
            i += 1
        elif line.startswith('deleted file mode'):
            if current_file:
                files[current_file]['status'] = 'deleted'
            i += 1
        elif line.startswith('+') and not line.startswith('+++'):
            # 简单的关键词提取逻辑
            if current_file == 'flare/configs/default_policy.yaml':
                for key in ['steps', 'batch_size', 'num_workers', 'prefetch_factor']:
                    if re.search(rf'^\+\s+{key}:', line):
                        files[current_file]['key_changes'].append(line.strip('+ '))
            i += 1
        else:
            i += 1

    # 分类汇总
    modified = [f for f, info in files.items() if info['status'] == 'modified']
    new_files = [f for f, info in files.items() if info['status'] == 'new']
    deleted = [f for f, info in files.items() if info['status'] == 'deleted']

    # 生成 Markdown 报告
    report = []
    report.append(f"# 自动化代码变更分析报告")
    report.append(f"\n> 分析源文件: `{diff_file}`")
    report.append(f"\n---")
    
    report.append(f"\n## 一、变更摘要")
    report.append(f"| 变更类型 | 文件数量 |")
    report.append(f"| :--- | :--- |")
    report.append(f"| 修改 (Modified) | {len(modified)} |")
    report.append(f"| 新增 (New) | {len(new_files)} |")
    report.append(f"| 删除 (Deleted) | {len(deleted)} |")
    report.append(f"| **总计** | **{len(files)}** |")

    if modified:
        report.append(f"\n## 二、核心文件修改详情")
        for f in sorted(modified):
            if files[f]['key_changes']:
                report.append(f"\n#### `{f}`")
                for change in files[f]['key_changes']:
                    report.append(f"- 检测到配置变更: `{change}`")
            else:
                report.append(f"- 修改了 `{f}`")

    if new_files:
        report.append(f"\n## 三、新增文件结构汇总")
        dirs = {}
        for f in sorted(new_files):
            parts = f.split('/')
            d = parts[0] if len(parts) > 1 else "根目录"
            if d not in dirs: dirs[d] = []
            dirs[d].append(f)
        
        for d, fs in dirs.items():
            report.append(f"\n### {d}/")
            for f in fs:
                report.append(f"- `{f}`")

    report.append(f"\n---")
    report.append(f"\n*报告由分析脚本自动生成于 2026-01-12*")

    return "\n".join(report)

if __name__ == "__main__":
    diff_path = sys.argv[1] if len(sys.argv) > 1 else "current_vs_dcc5e3d_no_latent.diff"
    output_path = "auto_report_v2.md"
    
    report_md = analyze_diff(diff_path)
    if report_md:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"报告已生成至: {output_path}")

