# To  run the code from the terminal, please write: streamlit run file_name.py

import streamlit as st
import google.generativeai as genai
import json
import os
import io
import re 

# Configuration

st.set_page_config(
    page_title="ATS-Friendly CV Generator",
    page_icon="📄",
    layout="wide"
)
# You can get api key from this link "https://aistudio.google.com/api-keys"
HARDCODED_KEY = "AIzaSyBH*******************************" 
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = HARDCODED_KEY


genai.configure(api_key=GOOGLE_API_KEY)


# The JSON schema we will force the model to output
# Simplified to a single markdown string, which maps better to a LaTeX doc
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "full_cv_markdown": {
            "type": "STRING",
            "description": "A single, complete CV formatted in Markdown. It must include the header (# Name), contact bar, and all sections (## Experience, ## Education, ## Projects, etc.) in a logical, single-column flow."
        },
        "missingInfo": {
            "type": "ARRAY",
            "items": { "type": "STRING" },
            "description": "A list of 3-5 critical questions for the user to fill in gaps (e.g., 'What were the dates for your role at [Company]?', 'I noticed you mentioned LinkedIn, what is the full URL?')."
        }
    },
    "required": ["full_cv_markdown", "missingInfo"]
}

# Few-Shot Examples
SYSTEM_PROMPT = """
You are an expert resume formatter. Your task is to take a user's rough CV text and re-format it into a specific, single-column Markdown structure, identical to the provided "gold standard" examples.
This single Markdown document will be used to generate a LaTeX file.
You MUST respond ONLY with the specified JSON format. Failure to use the exact JSON structure will break the application.

---
**GOLD STANDARD EXAMPLE 1 (KY_CV.pdf):**
# Kirellos Youssef
# NOTE 2: Contact info is a SINGLE horizontal bar.
Egypt | [kirellosyoussef21@ieee.org](mailto:kirellosyoussef21@ieee.org) | 01501701693 | [Portfolio](https://kirellosyoussef.github.io/) | [Linkedin](https://www.linkedin.com/in/kirellos-youssef/) | [GitHub](https://github.com/KirellosYoussef)

## Experience
**AI Intern, Samsung - Smart Village, Egypt**
*Aug 2025-Nov 2025*
* Strengthened foundations in math, statistics, and core Al concepts
* Applied preprocessing, visualization, and machine learning...

**Research Intern, IEEE EMBS - Pune, India**
*June 2025-July 2025*
* Applied machine learning and deep learning techniques...

## Technologies
# NOTE 3: Skills are a SINGLE bullet point per category.
* **Programming and Tools:** Python, C++, C, SQL, Git, GitHub, ROS, Linux...
* **AI and ML:** Machine Learning, Reinforcement Learning, Deep Learning...
* **Libraries:** NumPy, Pandas, Plotly, Dash, Matplotlib...
---
**GOLD STANDARD EXAMPLE 2 (Ali Ahmed Ali resume.pdf):**
# Ali Ahmed Ali
# NOTE 2: Contact info is a SINGLE horizontal bar.
Egypt | [aliahmedali78mohamed1977@gmail.com](mailto:aliahmedali78mohamed1977@gmail.com) | 01282920783 | [Portfolio](https://aliahmedali.github.io/) | [Linkedin](https://www.linkedin.com/in/ali-ahmed-ali-730b7028b/) | [GitHub](https://github.com/AliAhmedAli7) | [Kaggle](https://www.kaggle.com/kaggle/aliahmedali7)

## Experience
**Microsoft Machine Learning Engineer Trainee - Digital Egypt Pioneers Initiative (DEPI)**
*June 2025 - Dec 2025*
* Selected for a competitive 6-month national program...
* Gaining practical expertise in probabilities and statistics...

## Education
**Modern University for Technology and Information, B.Sc. in Computer Science**
*Sep 2024-Jul 2027*
GPA: 3.69/4.0

## Projects
**Smart Employee Attendance System**
*[link](https://github.com/AliAhmedAli7/Smart-Employee-Attendance-System)*
* Face Recognition-Based Check-In - Built a real-time attendance system...
* Automated Entry Logging - Captured entry/exit times automatically...

## Skills
# NOTE 3: Skills are a SINGLE bullet point per category.
* **Languages:** Python, C++, Java, SQL, HTML/CSS
* **Frameworks and Tools:** Pandas, NumPy, Matplotlib, Seaborn, Plotly, Dash, BeautifulSoup, Selenium, Scikit-learn, Streamlit, OpenCV, Excel, Web Scraping, Git, GitHub
* **AI/ML:** Data Cleaning, Data Visualization, Machine Learning (Supervised Unsupervised), Deep Learning, Probability Statistics, Linear Algebra
---
**YOUR TASK - FOLLOW THESE RULES STRICTLY:**
1.  **Parse User CV:** Read the user's provided "rush CV".
2.  **Extract All Information:** Find all pieces of data (name, contact, experience, education, skills, projects, etc.).
3.  **Generate Links:** Create full Markdown links (e.g., `[LinkedIn](...)`, `[email@domain.com](mailto:...)`).
4.  **Generate ONE Markdown Document:** Create a *single* `full_cv_markdown` string.
5.  **Format with Markdown:**
    * `# Name` (for header)
    * **Contact Bar:** Place *all* contact items (Email, Phone, LinkedIn, etc.) on **one single line**, separated by ` | `.
    * `## Section Title` (for section headings)
    * `**Job Title, Company**` (for bold text)
    * `*Dates*` (for italics)
    * `* Bullet point description` (for job/project bullet points)
    * **Skills/Technologies/Languages:**
        * Create **one bullet point per category**.
        * **Example:** `* **Programming Languages:** Python, C, C++`
6.  # NOTE 1: Do NOT use slashes ( / ) or backslashes ( \ ) as separators. Use commas (,) to separate items on the same line.
7.  **Identify Gaps:** Fill the `missingInfo` array with 3-5 critical questions for the user.
8.  **FINAL CHECK:** Ensure your *entire* response is *only* the valid JSON object described in the schema. Do not add any text before or after the JSON.
"""

# --- End of Modification ---


def generate_cv_from_text(user_input_text):
    """
    Sends the user's CV text to the Gemini API and gets a structured response.
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": JSON_SCHEMA,
                "temperature": 0.3, # Lowered temperature for more precise formatting
            },
            system_instruction=SYSTEM_PROMPT
        )
        
        prompt = f"Here is the user's CV: \n\n {user_input_text}"
        
        response = model.generate_content(prompt)
        
        # Parse the JSON string from the model's response
        response_json = json.loads(response.text)
        
        # Return all the new parts
        return (
            response_json.get("full_cv_markdown"),
            response_json.get("missingInfo")
        )

    except Exception as e:
        st.error(f"An error occurred while contacting the AI model: {e}")
        return None, None

# LaTeX Conversion
def markdown_to_latex(full_cv_markdown):
    """
    Converts the AI's single Markdown string into a runnable LaTeX (.tex) file
    based on the user's provided template.
    """
    
    # This is the full, correct preamble from the user's template
    preamble = r"""
\documentclass[10pt, letterpaper]{article}

% Packages:
\usepackage[
    ignoreheadfoot, % set margins without considering header and footer
    top=2 cm, % seperation between body and page edge from the top
    bottom=2 cm, % seperation between body and page edge from the bottom
    left=2 cm, % seperation between body and page edge from the left
    right=2 cm, % seperation between body and page edge from the right
    footskip=1.0 cm, % seperation between body and footer
    % showframe % for debugging 
]{geometry} % for adjusting page geometry
\usepackage{titlesec} % for customizing section titles
\usepackage{tabularx} % for making tables with fixed width columns
\usepackage{array} % tabularx requires this
\usepackage[dvipsnames]{xcolor} % for coloring text
\definecolor{primaryColor}{RGB}{0, 0, 0} % define primary color
\usepackage{enumitem} % for customizing lists
\usepackage{fontawesome5} % for using icons
\usepackage{amsmath} % for math
\usepackage[
    pdftitle={CV},
    pdfauthor={User},
    pdfcreator={LaTeX with RenderCV},
    colorlinks=true,
    urlcolor=primaryColor
]{hyperref} % for links, metadata and bookmarks
\usepackage[pscoord]{eso-pic} % for floating text on the page
\usepackage{calc} % for calculating lengths
\usepackage{bookmark} % for bookmarks
\usepackage{lastpage} % for getting the total number of pages
\usepackage{changepage} % for one column entries (adjustwidth environment)
\usepackage{paracol} % for two and three column entries
\usepackage{ifthen} % for conditional statements
\usepackage{needspace} % for avoiding page brake right after the section title
\usepackage{iftex} % check if engine is pdflatex, xetex or luatex

% Ensure that generate pdf is machine readable/ATS parsable:
\ifPDFTeX
    \input{glyphtounicode}
    \pdfgentounicode=1
    \usepackage[T1]{fontenc}
    \usepackage[utf8]{inputenc}
    \usepackage{lmodern}
\fi

\usepackage{charter}

% Some settings:
\raggedright
\AtBeginEnvironment{adjustwidth}{\partopsep0pt} % remove space before adjustwidth environment
\pagestyle{empty} % no header or footer
\setcounter{secnumdepth}{0} % no section numbering
\setlength{\parindent}{0pt} % no indentation
\setlength{\topskip}{0pt} % no top skip
\setlength{\columnsep}{0.15cm} % set column seperation
\pagenumbering{gobble} % no page numbering

\titleformat{\section}{\needspace{4\baselineskip}\bfseries\large}{}{0pt}{}[\vspace{1pt}\titlerule]

\titlespacing{\section}{
    % left space:
    -1pt
}{
    % top space:
    0.3 cm
}{
    % bottom space:
    0.2 cm
} % section title spacing

\renewcommand\labelitemi{$\vcenter{\hbox{\small$\bullet$}}$} % custom bullet points
\newenvironment{highlights}{
    \begin{itemize}[
        topsep=0.10 cm,
        parsep=0.10 cm,
        partopsep=0pt,
        itemsep=0pt,
        leftmargin=0 cm + 10pt
    ]
}{
    \end{itemize}
} % new environment for highlights


\newenvironment{highlightsforbulletentries}{
    \begin{itemize}[
        topsep=0.10 cm,
        parsep=0.10 cm,
        partopsep=0pt,
        itemsep=0pt,
        leftmargin=10pt
    ]
}{
    \end{itemize}
} % new environment for highlights for bullet entries

\newenvironment{onecolentry}{
    \begin{adjustwidth}{
        0 cm + 0.00001 cm
    }{
        0 cm + 0.00001 cm
    }
}{
    \end{adjustwidth}
} % new environment for one column entries

\newenvironment{twocolentry}[2][]{
    \onecolentry
    \def\secondColumn{#2}
    \setcolumnwidth{\fill, 4.5 cm}
    \begin{paracol}{2}
}{
    \switchcolumn \raggedleft \secondColumn
    \end{paracol}
    \endonecolentry
} % new environment for two column entries

\newenvironment{threecolentry}[3][]{
    \onecolentry
    \def\thirdColumn{#3}
    \setcolumnwidth{, \fill, 4.5 cm}
    \begin{paracol}{3}
    {\raggedright #2} \switchcolumn
}{
    \switchcolumn \raggedleft \thirdColumn
    \end{paracol}
    \endonecolentry
} % new environment for three column entries

\newenvironment{header}{
    \setlength{\topsep}{0pt}\par\kern\topsep\centering\linespread{1.5}
}{
    \par\kern\topsep
} % new environment for the header

\newcommand{\placelastupdatedtext}{% \placetextbox{<horizontal pos>}{<vertical pos>}{<stuff>}
  \AddToShipoutPictureFG*{% Add <stuff> to current page foreground
    \put(
        \LenToUnit{\paperwidth-2 cm-0 cm+0.05cm},
        \LenToUnit{\paperheight-1.0 cm}
    ){\vtop{{\null}\makebox[0pt][c]{
        \small\color{gray}\textit{Last updated in September 2024}\hspace{\widthof{Last updated in September 2024}}
    }}}%
  }%
}%

% save the original href command in a new command:
\let\hrefWithoutArrow\href
"""
#  Helper functions for parsing ---
    
    def escape_latex(text):
        """Escapes special LaTeX characters in a string."""
        if not text: return ""
        text = text.replace('&', r'\&')
        text = text.replace('%', r'\%')
        text = text.replace('$', r'\$')
        text = text.replace('#', r'\#')
        text = text.replace('_', r'\_')
        text = text.replace('{', r'\{')
        text = text.replace('}', r'\}')
        text = text.replace('~', r'\textasciitilde{}')
        text = text.replace('^', r'\textasciicircum{}')
        text = text.replace('\\', r'\textbackslash{}')
        return text

    def process_md_line(line):
            """Converts a single line of Markdown to LaTeX, handling links and styles."""

            line = re.sub(
                r'\[([^\]]+)\]\(([^)]+)\)', 
                lambda m: fr'\hrefWithoutArrow{{{escape_latex(m.group(2))}}}{{{escape_latex(m.group(1))}}}', 
                line
            )
            
            line = re.sub(
                r'\*\*(.*?)\*\*', 
                lambda m: fr'\textbf{{{escape_latex(m.group(1))}}}', 
                line
            )
            
            line = re.sub(
                r'\*(.*?)\*', 
                lambda m: fr'\textit{{{escape_latex(m.group(1))}}}', 
                line
            )

            if not any(cmd in line for cmd in [r'\href', r'\textbf', r'\textit']):
                 line = escape_latex(line)
                 
            return line

    # Document Body Generation
    latex_body = []
    lines = full_cv_markdown.split('\n')
    
    if not lines:
        return preamble + "\\begin{document}\n\\end{document}"

# 1. Process Header
    header_line = lines[0].lstrip('# ').strip()
    latex_body.append(r'\begin{header}')
    latex_body.append(fr'    \fontsize{{25 pt}}{{25 pt}}\selectfont {escape_latex(header_line)} \\') 
    latex_body.append(r'    \vspace{5 pt}')
    
    # 2. Process Contact Bar
    if len(lines) > 1:
        contact_line = lines[1]
        contact_parts = contact_line.split(' | ')
        latex_contact_parts = []
        for part in contact_parts:
            part = part.strip()
            # Use your updated process_md_line function here
            part_tex = process_md_line(part) 
            latex_contact_parts.append(fr'\mbox{{{part_tex}}}')
        
        latex_body.append(r'    \normalsize')
        # FIX 2: Add a space at the end of the joiner string
        latex_body.append(r'    \kern 5.0 pt \AND \kern 5.0 pt '.join(latex_contact_parts)) 
    
    latex_body.append(r'\end{header}')
    latex_body.append(r'\vspace{5 pt - 0.3 cm}')
    
    # 3. Process Main Body (Sections, Entries, etc.)
    i = 2 # Start after header (line 0) and contact bar (line 1)
        

    # position (the first '## Section'), and the 'else' block
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
            
        # 3.1. Handle Section
        if line.startswith('## '):
            title = process_md_line(line.lstrip('## '))
            latex_body.append(f"\\section{{{title}}}")
            i += 1
            continue

        # 3.2. Check for "Entry" (Title + Date + Bullets)
        is_entry = False
        if line.startswith('**'):
            if i + 1 < len(lines) and lines[i+1].strip().startswith('*') and not lines[i+1].strip().startswith('* '):
                is_entry = True

        if is_entry:
            title = process_md_line(line) 
            date = process_md_line(lines[i+1].strip())
            i += 2 # Consume title and date

            latex_body.append(f"\\begin{{twocolentry}}{{{date}}}")
            latex_body.append(f"    {title}")
            latex_body.append(f"\\end{{twocolentry}}")

            # Look ahead for bullets
            bullets = []
            if i < len(lines) and lines[i].strip().startswith('* '):
                while i < len(lines) and lines[i].strip().startswith('* '):
                    bullet_text = process_md_line(lines[i].strip().lstrip('* '))
                    bullets.append(f"    \\item {bullet_text}")
                    i += 1
            
            if bullets:
                latex_body.append(r"\vspace{0.10 cm}")
                latex_body.append(r"\begin{onecolentry}")
                latex_body.append(r"    \begin{highlights}")
                latex_body.extend(bullets)
                latex_body.append(r"    \end{highlights}")
                latex_body.append(r"\end{onecolentry}")

            latex_body.append(r"\vspace{0.2 cm}")
        
        # 3.3. Handle simple text/skills (not an entry)
        else:
            if line.startswith('* '):
                latex_body.append(r"\begin{onecolentry}")
                latex_body.append(r"    \begin{highlightsforbulletentries}")
                while i < len(lines) and lines[i].strip().startswith('* '):
                    bullet_text = process_md_line(lines[i].strip().lstrip('* '))
                    latex_body.append(f"    \\item {bullet_text}")
                    i += 1
                latex_body.append(r"    \end{highlightsforbulletentries}")
                latex_body.append(r"\end{onecolentry}")
            else:
                latex_body.append(r"\begin{onecolentry}")
                latex_body.append(f"    {process_md_line(line)}")
                latex_body.append(r"\end{onecolentry}")
                i += 1
    
    # Combine all parts 

    latex_doc = fr"""
{preamble}
\begin{{document}}

% --- Definitions from template body ---
\newcommand{{\AND}}{{\unskip
    \cleaders\copy\ANDbox\hskip\wd\ANDbox
    \ignorespaces
}}
\newsavebox\ANDbox
\sbox\ANDbox{{$|$}}
% --- End of Definitions ---

{'\n'.join(latex_body)}

\end{{document}}
"""
    return latex_doc
# End of LaTeX Function

# Streamlit UI

st.title("📄 AI-Powered CV Generator ")
st.subheader("Paste your 'rush' CV below and let AI re-format it like the 2-column template.")

# Initialize session state
if "full_cv_markdown" not in st.session_state:
    st.session_state.full_cv_markdown = ""
if "missing_info" not in st.session_state:
    st.session_state.missing_info = []

# Input Text Area
input_cv = st.text_area("Paste your CV here:", height=300, 
                        placeholder="e.g., John Doe - worked at google 2018. managed stuff. good at python. degree from state u...")

if st.button("✨ Generate Polished CV", type="primary"):
    if input_cv:
        with st.spinner("Analyzing and re-formatting your CV..."):
            full_cv_markdown, missing_info = generate_cv_from_text(input_cv)
            
            if full_cv_markdown and missing_info:
                st.session_state.full_cv_markdown = full_cv_markdown
                st.session_state.missing_info = missing_info
                st.success("CV generation complete!")
            else:
                st.error("Failed to generate CV. Please check the console for errors.")
    else:
        st.warning("Please paste your CV text into the box above.")

# Display Results

if st.session_state.full_cv_markdown:
    st.divider()
    
    # Main container for the CV preview and Suggestions
    cv_container, suggestions_container = st.columns([2.5, 1.5]) # Give CV more space
    
    with cv_container:
        st.header("Your New Formatted CV")
        
        # Display the single, continuous markdown string
        st.markdown(st.session_state.full_cv_markdown)
        
    with suggestions_container:
        st.header("🔍 Suggested Improvements")
        st.write("Answer these questions to make your CV even stronger:")
        
        if st.session_state.missing_info:
            for i, question in enumerate(st.session_state.missing_info):
                st.info(f"**{i+1}.** {question}")
        else:
            st.info("No immediate gaps found. Good job!")

    # Download Button Section ---
    st.divider()
    st.header("⬇️ Download Your CV")
    
    try:
        # Generate the LaTeX code
        latex_cv = markdown_to_latex(
            st.session_state.full_cv_markdown
        )
        
        # Display the single download button
        st.download_button(
            label="Download as .tex",
            data=latex_cv,
            file_name="Your_CV.tex",
            mime="text/latex",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"An error occurred while generating the LaTeX file: {e}")

st.markdown("Check out the [Overleaf site](https://www.overleaf.com/) to convert Latex code to PDF format!")