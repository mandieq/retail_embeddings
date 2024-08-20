# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mesop as me
import pandas as pd
import numpy as np
from io import StringIO
from ast import literal_eval

from google.cloud import storage
from vertexai.vision_models import MultiModalEmbeddingModel

# ======== Environment set up ========

BUCKET = "YOUR_BUCKET_NAME_HERE"
BUCKET_URI = f"gs://{BUCKET}/"
PROJECT_ID = "YOUR_PROJECT_NAME_HERE"  
LOCATION = "YOUR_REGION_HERE" 

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)
print(f"Using vertexai version: {vertexai.__version__}")

storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET)

embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# ======== State class ========

@me.stateclass
class State:
    selected_material: str = "" 
    product_df: pd.DataFrame | None = None
    uri_exists: bool = True
    search_term: str = ""
    sim_df: pd.DataFrame | None = None

# ======== Functions to handle state changes ========

def button_nav_load(e: me.ClickEvent):
    s = me.state(State)
    me.navigate("/")

def button_nav_search(e: me.ClickEvent):
    s = me.state(State)
    me.navigate("/search")

def on_blur_input_csv(e: me.InputBlurEvent):
    s = me.state(State)
    s.selected_material = e.value

def on_blur_input_search(e: me.InputBlurEvent):
    s = me.state(State)
    s.search_term = e.value

def on_click_load(e: me.ClickEvent):
    s = me.state(State)
    s.product_df = None 
    print(f"\nRunning load function")
    print(f"Input is {s.selected_material}")
    if not bucket.blob(f"{s.selected_material}/product_details.csv").exists():
        print(f"Error: {s.selected_material}/product_details.csv does not exist!")
        s.uri_exists = False
        return
    else:
        s.uri_exists = True
    data_str = bucket.blob(f"{s.selected_material}/product_details.csv").download_as_text()
    s.product_df = pd.read_csv(StringIO(data_str), converters={'image_embedding': literal_eval, 'text_embedding': literal_eval})

def on_click_search(e: me.ClickEvent):
    s = me.state(State)
    s.sim_df = None
    print(f"\nRunning search function")
    if s.search_term:
        print(f"Search term: {s.search_term}") 
        search_embedding = get_text_embedding(s.search_term)
        s.sim_df = get_similar_products(search_embedding, s.product_df)
    else:
        print("No search term entered")

# ======== Helper functions ========

def get_text_embedding(text):
    search_embed = embedding_model.get_embeddings(contextual_text=text).text_embedding
    return search_embed

def get_cosine_score(dataframe: pd.DataFrame, column_name: str, input_text_embd: np.ndarray) -> float:
    """
    Calculates the cosine similarity between the user query embedding and the dataframe embedding 
    for a specific column.

    Args:
        dataframe: The pandas DataFrame containing the data to compare against.
        column_name: The name of the column containing the embeddings to compare with.
        input_text_embd: The NumPy array representing the user query embedding.

    Returns:
        The cosine similarity score (rounded to two decimal places) between the user query 
        embedding and the dataframe embedding.
    """
    dot_product = np.dot(dataframe[column_name], input_text_embd)
    norm_a = np.linalg.norm(dataframe[column_name])
    norm_b = np.linalg.norm(input_text_embd)
    text_cosine_score = dot_product / (norm_a * norm_b)
    return round(text_cosine_score, 3)

def get_similar_products(query_emb, df):
    """
    Take an embedding for a query, scores against the dataframe of options and
    returns a dataframe showing the top three matches.

    Args:
        query_emb: The embedding for the query term
        df: The pandas DataFrame containing the data to compare against.

    Returns:
        A pandas DataFrame with the top three matches.
    """
    temp = df.copy()
    temp['cosine_score'] = temp.apply(lambda x: get_cosine_score(x, "image_embedding", query_emb), axis=1)
    temp = temp.sort_values('cosine_score', ascending=False).head(3)
    print("Top 3 results found:", temp)
    return temp

# ======== Page set up ========

MAIN_TITLE = "Retail embeddings demo"
BACKGROUND_COLOUR = "#f0f4f8"

# Load product details page 
LOAD_TITLE="Load product details"
@me.page(title=f"{MAIN_TITLE}", path="/")
def page_load():
    s = me.state(State)
    with me.box(style=STYLE_BACK):
        with me.box(style=STYLE_BOX_HOLDING):
            me.markdown(f"#{MAIN_TITLE}")
            with me.box(style=STYLE_TITLE):
                me.markdown(f"##{LOAD_TITLE}")
                me.button(label=SEARCH_TITLE, type="raised", on_click=button_nav_search, style=STYLE_BUTTON)
        with me.box(style=STYLE_BOX_HOLDING):
            with me.box(style=STYLE_BOX_WHITE):
                me.markdown(f"Using GCS bucket: **{BUCKET_URI}**")
                with me.box(style=me.Style(display="flex", flex_direction="column", flex_grow=2)):
                    me.input(label="Product set folder (without trailing slash)", on_blur=on_blur_input_csv, value=s.selected_material, style=me.Style(width="100%"))
        with me.box(style=STYLE_BOX_HOLDING):
            me.button(label="Load product data", type="raised", on_click=on_click_load)

        if len(s.selected_material) > 0:
            if s.uri_exists == False:
                with me.box(style=STYLE_BOX_HOLDING):
                    with me.box(style=STYLE_BOX_WHITE):
                        me.markdown(f"{BUCKET_URI}{s.selected_material}/product_details.csv not found", style=me.Style(color="red"))
            elif s.product_df is not None:
                with me.box(style=STYLE_BOX_HOLDING):
                    with me.box(style=STYLE_BOX_WHITE_IMAGES):
                        for img in list(s.product_df.rel_image_uri)[:5]:
                            img_url = f"https://storage.cloud.google.com/{BUCKET}/{img.replace('.png', '_sm.png')}"
                            me.image(src=img_url, style=me.Style(width="100%", border_radius=10))
                    with me.box(style=STYLE_BOX_WHITE_IMAGES):
                        for img in list(s.product_df.rel_image_uri)[5:]:
                            img_url = f"https://storage.cloud.google.com/{BUCKET}/{img.replace('.png', '_sm.png')}"
                            me.image(src=img_url, style=me.Style(width="100%", border_radius=10))
                with me.box(style=STYLE_BOX_HOLDING):
                    with me.box(style=STYLE_BOX_WHITE):
                        # me.markdown(f"Dataset selected: **{s.selected_material}**")
                        me.table(s.product_df[['name', 'brand', 'colour or flavour', 'sku', 'department']])
            me.box(style=me.Style(height=32))
    
# Search page   
SEARCH_TITLE="Search using embeddings"
@me.page(title=f"{SEARCH_TITLE}", path="/search")
def page_search():
    s = me.state(State)
    with me.box(style=STYLE_BACK):
        with me.box(style=STYLE_BOX_HOLDING):
            me.markdown(f"#{MAIN_TITLE}")
            with me.box(style=STYLE_TITLE):
                me.markdown(f"##{SEARCH_TITLE}")
                me.button(label=f"Back to {LOAD_TITLE.lower()}", type="raised", on_click=button_nav_load, style=STYLE_BUTTON)

        if s.product_df is None:
            with me.box(style=STYLE_BOX_HOLDING):
                with me.box(style=STYLE_BOX_WHITE):
                    me.markdown("Choose a product set to begin")

        if s.product_df is not None:
            with me.box(style=STYLE_BOX_HOLDING):
                with me.box(style=STYLE_BOX_WHITE):
                    me.markdown("Enter a search term to find semantically similar images")
                    with me.box(style=me.Style(display="flex", flex_direction="column", flex_grow=2)):
                        me.input(label="Search term", on_blur=on_blur_input_search, value=s.search_term, style=me.Style(width="100%"))
                    me.button(label="Search embeddings", type="raised", on_click=on_click_search)

        if s.sim_df is not None:
            with me.box(style=STYLE_BOX_HOLDING):
                with me.box(style=STYLE_BOX_WHITE):
                    with me.box(style=me.Style(display="flex", justify_content="center", text_align="center")):
                        for index, row in s.sim_df.iterrows():
                            with me.box():
                                me.markdown(f"**{row['name']}**")
                                me.markdown(f"Score: {row.cosine_score:.3f}")
                                img_url = f"https://storage.cloud.google.com/{BUCKET}/{row.rel_image_uri.replace('.png', '_sm.png')}"
                                me.image(src=img_url, style=me.Style(width="80%", border_radius=10))
    me.box(style=me.Style(height=32))


# ======== Formatting ========

STYLE_BACK = me.Style(
    background=BACKGROUND_COLOUR,
    height="100%",
    overflow_y="scroll",
)

STYLE_BOX_HOLDING = me.Style(
    background="#f0f4f8",
    margin=me.Margin(left="auto", right="auto"),
    padding=me.Padding(top=24, left=24, right=24, bottom=0),
    width="min(1024px, 100%)",
    display="flex",
    flex_direction="column",
)

STYLE_TITLE = me.Style(
    display="flex", 
    flex_direction="row", 
    justify_content="space-between", 
    padding=me.Padding.all(12)
)

STYLE_BUTTON = me.Style(
    margin=me.Margin(left=12, top=12)
)

STYLE_BOX_WHITE = me.Style(
    flex_basis="max(480px, calc(50% - 48px))",
    background="#fff",
    border_radius=12,
    box_shadow=(
    "0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f"
    ),
    padding=me.Padding(top=16, left=16, right=16, bottom=16),
    display="flex",
    flex_direction="column",
)

STYLE_BOX_WHITE_IMAGES = me.Style(
    flex_basis="max(480px, calc(50% - 48px))",
    # background="#fff",
    # border_radius=12,
    # box_shadow=(
    # "0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f"
    # ),
    padding=me.Padding(top=4, left=4, right=4, bottom=4),
    display="flex",
    flex_direction="row",
    gap=8,
    # flex_wrap="wrap"
)
