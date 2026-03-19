import re
import os
import sys

def tex_to_md(tex_content):
    # Extract title from preamble if available
    title_match = re.search(r'\\title\{(.*?)\}', tex_content, re.DOTALL)
    title = title_match.group(1) if title_match else "Untitled"

    # Extract the main document body
    body_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', tex_content, re.DOTALL)
    if not body_match:
        return "Error: Could not find \begin{document}...\end{document}"

    content = body_match.group(1).strip()

    # Replace \maketitle with # Title
    content = re.sub(r'\\maketitle', f'# {title}\n', content)

    # Replace \section{...} with # ...
    content = re.sub(r'\\section\{(.*?)\}', r'# \1', content)

    # Replace \subsection{...} with ## ...
    content = re.sub(r'\\subsection\{(.*?)\}', r'## \1', content)

    # Replace \subsubsection{...} with ### ...
    content = re.sub(r'\\subsubsection\{(.*?)\}', r'### \1', content)

    # Handle simple math conversions:
    # 1. Block math \[ ... \] to $$ ... $$
    content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)

    # 2. Inline math \( ... \) to $ ... $
    content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content, flags=re.DOTALL)

    # 3. Environment math \begin{equation} ... \end{equation} to $$ ... $$
    content = re.sub(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', r'$$\1$$', content, flags=re.DOTALL)

    # 4. Environment math \begin{align} ... \end{align} to $$ ... $$
    content = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', r'$$\1$$', content, flags=re.DOTALL)

    # Remove some common LaTeX commands that don't translate well to MD
    content = re.sub(r'\\label\{.*?\}', '', content)
    content = re.sub(r'\\cite\{.*?\}', '[Ref]', content)
    content = re.sub(r'\\ref\{.*?\}', '[Link]', content)

    # Replace multiple newlines with a single double-newline for markdown readability
    content = re.sub(r'\n\s*\n', '\n\n', content)

    return content.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python tex_to_md.py <input.tex>")
        return

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Define the output directory
    output_dir = "notes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the output file path in the knowledge_base directory
    base_name = os.path.basename(input_file)
    output_filename = os.path.splitext(base_name)[0] + ".md"
    output_file = os.path.join(output_dir, output_filename)

    with open(input_file, 'r', encoding='utf-8') as f:
        tex_data = f.read()

    md_data = tex_to_md(tex_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_data)

    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    main()
