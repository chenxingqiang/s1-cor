#!/usr/bin/env python3
"""Upload CoR paper and resources to Hugging Face."""

import os
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from pathlib import Path

def main():
    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/CoR-Chain-of-Reward"
    
    print(f"📤 Uploading to: https://huggingface.co/spaces/{repo_id}")
    
    # Create Space repository
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
            private=False,
        )
        print(f"✅ Created/verified Space: {repo_id}")
    except Exception as e:
        print(f"⚠️ Repo creation: {e}")
    
    # Upload README (model card)
    print("\n📄 Uploading README...")
    upload_file(
        path_or_fileobj="hf_paper_card.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    
    # Upload paper PDF
    print("📄 Uploading paper PDF...")
    upload_file(
        path_or_fileobj="paper/main.pdf",
        path_in_repo="paper.pdf",
        repo_id=repo_id,
        repo_type="space",
    )
    
    # Upload figures
    figures_dir = Path("paper/figures")
    for fig in figures_dir.glob("*.pdf"):
        print(f"📊 Uploading {fig.name}...")
        upload_file(
            path_or_fileobj=str(fig),
            path_in_repo=f"figures/{fig.name}",
            repo_id=repo_id,
            repo_type="space",
        )
    
    # Upload drawio files
    for fig in figures_dir.glob("*.drawio"):
        print(f"📐 Uploading {fig.name}...")
        upload_file(
            path_or_fileobj=str(fig),
            path_in_repo=f"figures/{fig.name}",
            repo_id=repo_id,
            repo_type="space",
        )
    
    # Create index.html for static Space
    index_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CoR: Chain of Reward</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        .links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .links a {
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .links a:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .result-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .result-table th, .result-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }
        .result-table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .result-table tr:last-child {
            background: #e8f4fd;
            font-weight: 600;
        }
        .highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        .formula {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            margin: 20px 0;
        }
        .section {
            margin: 30px 0;
        }
        h2 {
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .embed-container {
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 CoR: Chain of Reward</h1>
        <p class="subtitle">Endogenous Self-Evaluation for Sample-Efficient Reasoning</p>
        
        <div class="links">
            <a href="paper.pdf" target="_blank">📄 Paper PDF</a>
            <a href="https://github.com/chenxingqiang/s1-cor" target="_blank">💻 GitHub</a>
            <a href="https://huggingface.co/datasets/xingqiang/s1K-cor-deepseek" target="_blank">📊 Dataset</a>
        </div>
        
        <div class="section">
            <h2>Key Insight</h2>
            <p>Instead of only rewarding correct answers, we reward the model for <strong>accurate self-assessment</strong>. The model generates self-ratings during reasoning and learns to calibrate its confidence—enabling genuine self-correction.</p>
        </div>
        
        <div class="section">
            <h2>Core Formula</h2>
            <div class="formula">
                R(c) = R<sub>ext</sub> + λ·R<sub>int</sub> + μ·R<sub>improve</sub> + ν·R<sub>converge</sub>
            </div>
        </div>
        
        <div class="section">
            <h2>Results: <span class="highlight">800× Sample Efficiency</span></h2>
            <table class="result-table">
                <tr>
                    <th>Model</th>
                    <th>Samples</th>
                    <th>AIME24</th>
                    <th>MATH500</th>
                    <th>GPQA</th>
                </tr>
                <tr>
                    <td>o1-preview</td>
                    <td>N.A.</td>
                    <td>44.6</td>
                    <td>85.5</td>
                    <td>73.3</td>
                </tr>
                <tr>
                    <td>r1-distill</td>
                    <td>800K</td>
                    <td>72.6</td>
                    <td>94.3</td>
                    <td>62.1</td>
                </tr>
                <tr>
                    <td>CoR-32B (Ours)</td>
                    <td>1K</td>
                    <td>56.7</td>
                    <td>93.0</td>
                    <td>59.6</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Paper</h2>
            <iframe src="paper.pdf" class="embed-container"></iframe>
        </div>
    </div>
</body>
</html>"""
    
    print("🌐 Uploading index.html...")
    with open("/tmp/index.html", "w") as f:
        f.write(index_html)
    upload_file(
        path_or_fileobj="/tmp/index.html",
        path_in_repo="index.html",
        repo_id=repo_id,
        repo_type="space",
    )
    
    print(f"\n🎉 Upload complete!")
    print(f"🔗 View at: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
