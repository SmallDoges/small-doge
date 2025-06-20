# Copyright 2024 The SmallDoge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Theme and logo utilities for SmallDoge WebUI frontend.
Provides functions for generating HTML and CSS for the WebUI theme.
"""

import base64
from pathlib import Path

def get_logo_html():
    """Returns centered HTML code for doge_look.gif with base64 encoding"""
    gif_path = Path(__file__).parent / "assets" / "doge_look.gif"
    with open(gif_path, "rb") as f:
        doge_look_base64 = base64.b64encode(f.read()).decode()
    return f'''
    <div style="text-align:center;margin-bottom:16px;">
      <img src="data:image/gif;base64,{doge_look_base64}" style="width:120px;height:auto;"/>
    </div>
    '''

def get_theme_css():
    """Returns custom CSS styles for the theme"""
    return """
    /* Global styles */
    :root {
        --primary-color: #d97706;  /* Amber */
        --text-color: #4b3f2a;     /* Deep brown */
        --bg-color: #fef3c7;       /* Light yellow background */
    }

    /* Chat message styles */
    .message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        max-width: 80%;
    }

    .user-message {
        background-color: var(--primary-color);
        color: white;
        margin-left: auto;
    }

    .bot-message {
        background-color: var(--bg-color);
        color: var(--text-color);
        margin-right: auto;
    }

    /* Input field styles */
    .message-input {
        border: 2px solid var(--primary-color);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 1rem 0;
        width: 100%;
    }

    /* Button styles */
    .send-button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .send-button:hover {
        background-color: #b45309;
    }

    /* Discovery panel styles */
    .discovery-panel {
        background-color: var(--bg-color);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }

    .discovery-panel h3 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    /* Doge animation container styles */
    #doge-run-container {
        background: transparent;
        overflow: hidden;
        margin: 8px 0;
    }
    """

def get_custom_css():
    """Returns global theme CSS with Doge style, Discovery columns and chat bubble alignment"""
    return """
    body {
        background: #fffbe6;
        color: #4b3f2a;
        font-family: 'Comic Sans MS', 'Arial', sans-serif;
    }
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto;
        background: #fffbe6;
    }
    /* Header title styles */
    .header-row {
        margin-bottom: 20px;
        padding: 12px;
        background: linear-gradient(135deg, #fffbe6 0%, #fff8c5 100%);
        border-radius: 16px;
        border: 2px solid #f7d774;
    }
    .main-title {
        text-align: center;
    }
    .main-title h1 {
        margin: 0;
        color: #d97706;
        font-size: 2.2em;
        font-family: 'Comic Sans MS', 'Arial', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .main-title p {
        margin: 4px 0 0;
        color: #4b3f2a;
        font-size: 1.1em;
    }
    /* Main layout styles */
    .main-row {
        gap: 20px;
        align-items: stretch;
    }
    /* Sidebar styles */
    .sidebar {
        background: #fffbe6;
        border-radius: 16px;
        border: 2px solid #f7d774;
        padding: 0;
        height: 100%;
        min-height: 700px;
    }
    .sidebar-content {
        padding: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    .sidebar h3 {
        color: #d97706;
        margin: 16px 0 8px;
    }
    /* Chat area styles */
    .chat-container {
        border-radius: 16px;
        border: 2px solid #f7d774;
        background: #fffbe6;
        box-shadow: 0 4px 16px #f7d77433;
        padding: 16px;
        min-height: 700px;
        display: flex;
        flex-direction: column;
    }
    .chat-title h2 {
        font-family: 'Comic Sans MS', 'Arial', sans-serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        color: #d97706;
    }

    /* Input area layout */
    .input-row {
        gap: 12px;
        margin-top: auto;
        padding-top: 12px;
        align-items: flex-end !important;
    }
    .message-input {
        border-radius: 12px;
        border: 1.5px solid #f7d774;
        background: #ffffff;
        padding: 8px 16px;
        margin: 0;
        font-size: 1.1em;
        width: 100%;
        box-sizing: border-box;
    }
    .button-column {
        display: flex;
        flex-direction: column;
        gap: 8px;
        justify-content: flex-end;
        height: 100%;
        padding: 0;
    }
    .send-button, .cancel-button {
        border-radius: 12px;
        border: none;
        font-weight: 700;
        padding: 12px 24px;
        font-size: 1.1em;
        height: auto;
        min-height: 48px;
        width: 100%;
        transition: all 0.2s ease;
        margin: 0;
    }
    .send-button {
        background: linear-gradient(135deg, #ffe066 0%, #f7d774 100%);
        color: #4b3f2a;
    }
    .send-button:hover {
        background: #ffe066;
        color: #d97706;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .cancel-button {
        background: linear-gradient(135deg, #fecaca 0%, #f87171 100%);
        color: #ffffff;
    }
    .cancel-button:hover {
        background: #fecaca;
        color: #dc2626;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Chat message styles */
    .chat-messages {
        color: inherit !important;
        flex-grow: 1;
        margin-bottom: 12px;
    }
    .chat-messages .message {
        color: inherit !important;
    }
    .chat-messages .user-message {
        color: inherit !important;
    }
    .chat-messages .bot-message {
        color: inherit !important;
    }
    .chat-messages p, 
    .chat-messages span, 
    .chat-messages div {
        color: inherit !important;
    }

    /* Fix Gradio default styles */
    .gradio-container .gr-form {
        border: none !important;
        background: transparent !important;
    }
    .gradio-container .gr-input,
    .gradio-container .gr-textarea {
        border: 1.5px solid #f7d774 !important;
        background: #ffffff !important;
    }
    .gradio-container .gr-button {
        border: none !important;
    }
    .gradio-container .gr-button-primary {
        background: linear-gradient(135deg, #ffe066 0%, #f7d774 100%) !important;
        color: #4b3f2a !important;
    }
    .gradio-container .gr-button-secondary {
        background: #ffffff !important;
        border: 1.5px solid #f7d774 !important;
        color: #4b3f2a !important;
    }
    .gradio-container .gr-button-stop {
        background: linear-gradient(135deg, #fecaca 0%, #f87171 100%) !important;
        color: #ffffff !important;
    }

    .model-info {
        background: #ffffff;
        border: 1.5px solid #f7d774;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        color: #4b3f2a;
    }
    /* Discovery area styles */
    .gradio-container [class*="discovery-panel"],
    #discovery-panel {
        background: #fffbe6;
        border: 2px solid #f7d774;
        border-radius: 16px;
        padding: 18px;
        color: #4b3f2a;
        box-shadow: 0 2px 8px #f7d77422;
        margin-top: 20px;
    }

    /* Title and tag styles */
    .gradio-container [class*="discovery-panel"] h3,
    .gradio-container [class*="discovery-panel"] h4,
    .gradio-container [class*="discovery-panel"] label,
    #discovery-panel h3,
    #discovery-panel h4,
    #discovery-panel label {
        color: #d97706 !important;
        font-weight: 600 !important;
    }

    /* Markdown title styles */
    .gradio-container .markdown-style h3,
    .gradio-container .markdown-style h4 {
        color: #d97706 !important;
        font-weight: 600 !important;
    }

    /* Search results styles */
    .gradio-container .search-results,
    .gradio-container .model-result {
        color: #4b3f2a !important;
    }

    /* Search results title and count */
    .gradio-container .search-results h4,
    .gradio-container .search-results strong {
        color: #d97706 !important;
        font-weight: 600 !important;
    }

    /* Search results card styles */
    .gradio-container .model-result {
        background: #ffffff !important;
        border: 1.5px solid #f7d774 !important;
        margin: 8px 0 !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }

    /* Search results text styles */
    .gradio-container .model-result strong,
    .gradio-container .model-result b {
        color: #d97706 !important;
        font-weight: 600 !important;
    }

    .gradio-container .model-result small {
        color: #4b3f2a !important;
        opacity: 0.8 !important;
    }

    /* HTML content styles */
    .gradio-container .prose h1,
    .gradio-container .prose h2,
    .gradio-container .prose h3,
    .gradio-container .prose h4,
    .gradio-container .prose h5,
    .gradio-container .prose h6 {
        color: #d97706 !important;
        font-weight: 600 !important;
    }

    .gradio-container .prose p,
    .gradio-container .prose span,
    .gradio-container .prose div {
        color: #4b3f2a !important;
    }

    .gradio-container .prose strong,
    .gradio-container .prose b {
        color: #d97706 !important;
        font-weight: 600 !important;
    }

    /* Tag and search related styles */
    .gradio-container [data-testid*="tag"],
    .gradio-container .quick-tags,
    .gradio-container .popular-tags,
    #tag-based-search {
        color: #4b3f2a !important;
        font-weight: 500 !important;
    }

    /* Search field and input field styles */
    .gradio-container input[type="text"],
    .gradio-container input[type="search"],
    .gradio-container textarea {
        background: #ffffff !important;
        color: #4b3f2a !important;
        border: 1.5px solid #f7d774 !important;
    }

    /* Button styles */
    .gradio-container button:not([class*="cancel"]) {
        background: #ffe066 !important;
        color: #4b3f2a !important;
        border: 1.5px solid #f7d774 !important;
        font-weight: 500 !important;
    }

    .gradio-container button:not([class*="cancel"]):hover {
        background: #f7d774 !important;
        color: #4b3f2a !important;
    }

    /* Dropdown menu styles */
    .gradio-container select,
    .gradio-container .select {
        background: #ffffff !important;
        color: #4b3f2a !important;
        border: 1.5px solid #f7d774 !important;
    }

    /* Tag styles */
    .gradio-container .tag {
        background: #ffe066 !important;
        color: #4b3f2a !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
        margin: 2px !important;
        display: inline-block !important;
    }

    /* Strong text styles */
    .gradio-container strong,
    .gradio-container b {
        color: #d97706 !important;
    }

    /* Link styles */
    .gradio-container a {
        color: #d97706 !important;
        text-decoration: none !important;
    }

    .gradio-container a:hover {
        text-decoration: underline !important;
    }

    /* Chat bubble styles */
    .gr-chat-message {
        max-width: 70%;
        margin: 8px;
        padding: 12px 18px;
        border-radius: 16px;
        font-size: 1.1em;
        word-break: break-word;
        line-height: 1.5;
    }
    .gr-chat-message.user, .gr-chat-message:nth-child(even) {
        background: #ffe066;
        color: #4b3f2a;
        margin-left: auto;
        border-radius: 16px 16px 0 16px;
        text-align: right;
        border: 1.5px solid #f7d774;
    }
    .gr-chat-message.ai, .gr-chat-message:nth-child(odd) {
        background: #ffffff;
        color: #4b3f2a;
        margin-right: auto;
        border-radius: 16px 16px 16px 0;
        text-align: left;
        border: 1.5px solid #f7d774;
    }
    /* Tag text styles */
    label span {
        color: #4b3f2a !important;
    }
    /* Special button styles */
    button[variant="stop"] {
        background: #fecaca !important;
        color: #dc2626 !important;
        border-color: #f87171 !important;
    }
    button[variant="stop"]:hover {
        background: #f87171 !important;
        color: #ffffff !important;
    }

    /* Performance metrics display area styles */
    .performance-stats {
        background: rgba(255, 255, 255, 0.9);
        border: 1.5px solid #f7d774;
        border-radius: 12px;
        padding: 8px;
        margin: 8px 0;
        backdrop-filter: blur(4px);
    }
    
    .stat-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 12px;
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stat-label {
        color: #4b3f2a;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .stat-value {
        color: #d97706;
        font-weight: 600;
        font-size: 0.9em;
        font-family: monospace;
    }
    """ 