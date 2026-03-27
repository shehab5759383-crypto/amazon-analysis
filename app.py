import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title='Amazon Data Analysis', layout='wide')

# ============================================================
# LOAD & CLEAN DATA
# ============================================================
@st.cache_data
def clean_data(df):
    categories = {
        'Mobile & Accessories': ['phone', 'mobile', 'iphone', 'samsung', 'xiaomi', 'case', 'charger', 'screen protector', 'smartphone'],
        'Laptops & Computers':  ['laptop', 'computer', 'pc', 'macbook', 'notebook', 'dell', 'hp', 'lenovo', 'asus'],
        'Audio':                ['headphone', 'earphone', 'earbud', 'speaker', 'airpod', 'headset', 'audio', 'sound'],
        'TV & Video':           ['tv', 'television', 'monitor', 'display', '4k', 'hdmi', 'projector'],
        'Gaming':               ['gaming', 'game', 'xbox', 'playstation', 'ps5', 'ps4', 'controller', 'joystick'],
        'Cameras':              ['camera', 'lens', 'tripod', 'gopro', 'dslr', 'webcam'],
        'Smart Home':           ['smart', 'alexa', 'echo', 'wifi', 'router', 'bulb', 'plug'],
        'Wearables':            ['watch', 'fitbit', 'smartwatch', 'band', 'tracker'],
        'Storage':              ['ssd', 'hard drive', 'usb', 'flash', 'memory', 'sd card'],
        'Cables & Adapters':    ['cable', 'adapter', 'hub', 'connector', 'converter'],
    }
    def classify_product(title):
        if pd.isna(title): return 'Other'
        title_lower = title.lower()
        for category, keywords in categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        return 'Other'
    df['category'] = df['title'].apply(classify_product)

    df['rating'] = df['rating'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    df['bought_in_last_month'] = (
        df['bought_in_last_month']
        .str.replace('k+ bought in past month', '', regex=False)
        .str.replace('+ bought in past month', '', regex=False)
        .str.replace('K', '000', regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
    )
    df['listed_price'] = df['listed_price'].astype(str).str.replace('$', '').str.replace('No Discount', '0').str.replace(',', '').astype(float)
    df['number_of_reviews'] = df['number_of_reviews'].astype(str).str.replace(',', '').astype(float)
    df['current/discounted_price'] = df['current/discounted_price'].astype(str).str.replace(',', '').astype(float)
    df['is_couponed'] = pd.to_numeric(df['is_couponed'].astype(str).str.replace(r'[^0-9.]', '', regex=True).replace('', '0'), errors='coerce')
    df['price_on_variant'] = df['price_on_variant'].astype(str).str.replace(r'[^0-9.]', '', regex=True).replace('', '0').astype(float)

    df['delivery_details'] = df['delivery_details'].fillna('Not Available')
    df['is_free_delivery'] = df['delivery_details'].str.contains('FREE', case=False, na=False)
    df['has_fastest_delivery'] = df['delivery_details'].str.contains('fastest', case=False, na=False)

    if 'sustainability_badges' in df.columns:
        df.drop(columns=['sustainability_badges'], inplace=True)
    df.dropna(subset=['product_url'], inplace=True)
    df['buy_box_availability'] = df['buy_box_availability'].fillna('Not Available')

    df = df[(df['is_couponed'] >= 0) & (df['is_couponed'] <= 100)]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df['estimated_revenue'] = df['current/discounted_price'] * df['bought_in_last_month']
    df['discount_pct'] = ((df['listed_price'] - df['current/discounted_price']) / df['listed_price']) * 100
    df['discount_pct'] = df['discount_pct'].clip(lower=0)
    df['is_sponsored_enc'] = (df['is_sponsored'] == 'Sponsored').astype(int)
    df['is_free_delivery_enc'] = df['is_free_delivery'].astype(int)
    df['is_best_seller_enc'] = (df['is_best_seller'] == 'Best Seller').astype(int)
    df['price_range'] = pd.cut(df['current/discounted_price'],
                                bins=[0, 20, 50, 100, 200, 500, 99999],
                                labels=['$0-20', '$20-50', '$50-100', '$100-200', '$200-500', '$500+'])
    df['reviews_range'] = pd.cut(df['number_of_reviews'],
                                  bins=[0, 100, 500, 1000, 5000, 99999999],
                                  labels=['0-100', '100-500', '500-1k', '1k-5k', '5k+'])
    df['discount_range'] = pd.cut(df['is_couponed'],
                                   bins=[-1, 0, 10, 30, 60, 100],
                                   labels=['No Discount', '1-10%', '10-30%', '30-60%', '60%+'])
    df['rating_group'] = pd.cut(df['rating'],
                                 bins=[0, 3.5, 4.0, 4.5, 5.0],
                                 labels=['< 3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0'])
    return df


# ============================================================
# DATA LOADING — supports both local file and uploader
# ============================================================
LOCAL_CSV = 'amazon_products_sales_data_uncleaned.csv'

df = None

if os.path.exists(LOCAL_CSV):
    # If CSV is bundled alongside app.py in the repo
    df = clean_data(pd.read_csv(LOCAL_CSV))
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload amazon_products_sales_data_uncleaned.csv",
        type=["csv"]
    )
    if uploaded_file is not None:
        df = clean_data(pd.read_csv(uploaded_file))
    else:
        st.title(" Amazon Data Analysis")
        st.warning(
            "**No data found.**\n\n"
            "To fix this, do **one** of the following:\n\n"
            "1.  **Recommended**: Add `amazon_products_sales_data_uncleaned.csv` to your GitHub repo next to `app.py`.\n"
            "2.  Use the **Upload Data** button in the sidebar to upload the CSV manually each session."
        )
        st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title(" Amazon Analysis")
    st.markdown("---")
    page = st.radio(' Select Page', [
        '1 - Overview',
        '2 - Distributions',
        '3 - Revenue & Categories',
        '4 - What Drives Sales?',
        '5 - Opportunities'
    ])

    st.markdown("---")
    st.header(' Filters')
    selected_category = st.multiselect('Category', options=df['category'].unique(), default=df['category'].unique())
    price_min, price_max = st.slider('Price Range ($)',
                                      float(df['current/discounted_price'].min()),
                                      float(df['current/discounted_price'].max()),
                                      (0.0, 200.0))

fdf = df[
    (df['category'].isin(selected_category)) &
    (df['current/discounted_price'] >= price_min) &
    (df['current/discounted_price'] <= price_max)
]

# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================
if page == '1 - Overview':
    st.title(" Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", f"{len(fdf):,}")
    col2.metric("Avg Rating", f"{fdf['rating'].mean():.2f}")
    col3.metric("Avg Price", f"${fdf['current/discounted_price'].mean():.2f}")
    col4.metric("Avg Revenue", f"${fdf['estimated_revenue'].mean():,.0f}")

    st.markdown("---")
    tab1, tab2 = st.tabs([' Data Table', ' Categorical Distributions'])

    with tab1:
        st.dataframe(fdf[['title', 'category', 'rating', 'current/discounted_price',
                           'bought_in_last_month', 'estimated_revenue', 'is_sponsored', 'is_best_seller']].reset_index(drop=True))

    with tab2:
        exclude_cols = ['title', 'product_url', 'delivery_details', 'collected_at', 'image_url']
        cat_cols = [c for c in fdf.select_dtypes(include='object').columns if c not in exclude_cols]
        selected_cat = st.selectbox("Select Column", cat_cols)
        fig = px.pie(fdf, names=selected_cat, title=f'Distribution of {selected_cat}')
        st.plotly_chart(fig, use_container_width=True, key="pie_cat")

# ============================================================
# PAGE 2 - DISTRIBUTIONS
# ============================================================
elif page == '2 - Distributions':
    st.title(" Numeric Distributions")

    numeric_cols = ['rating', 'number_of_reviews', 'bought_in_last_month',
                    'estimated_revenue', 'discount_pct', 'current/discounted_price']

    col1, col2 = st.columns(2)
    for i, col in enumerate(numeric_cols):
        fig = px.histogram(fdf, x=col, title=f'Distribution of {col}')
        if i % 2 == 0:
            col1.plotly_chart(fig, use_container_width=True, key=f"dist_{col}")
        else:
            col2.plotly_chart(fig, use_container_width=True, key=f"dist_{col}")

    st.markdown("---")
    st.subheader(" Insights")
    st.info("""
    **1. Rating Distribution**
    Most products have ratings between 4.0 and 4.8, with almost nothing below 3.5.
    This doesn't mean all products are genuinely good — it reflects survivorship bias.
    Amazon's algorithm surfaces successful products, so the scraper collected mostly high-rated items.
    """)
    st.info("""
    **2. Price Distribution**
    Most products are low-priced, but there are extreme outliers reaching $2,000+.
    This suggests a two-tier market: affordable everyday products for the mass market,
    and premium products targeting a specific audience.
    """)

# ============================================================
# PAGE 3 - REVENUE & CATEGORIES
# ============================================================
elif page == '3 - Revenue & Categories':
    st.title(" Revenue & Categories")

    st.subheader(" Top 10 Products by Revenue")
    top10 = fdf[['title', 'current/discounted_price', 'bought_in_last_month', 'estimated_revenue']]\
            .sort_values('estimated_revenue', ascending=False).head(10)
    fig = px.bar(top10, x='estimated_revenue', y='title', orientation='h',
                 title='Top 10 Products by Estimated Revenue',
                 labels={'estimated_revenue': 'Revenue ($)', 'title': 'Product'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True, key="top10_bar")

    st.markdown("---")

    st.subheader(" Revenue by Category")
    cat_revenue = fdf.groupby('category').agg(
        total_revenue=('estimated_revenue', 'sum'),
        avg_revenue=('estimated_revenue', 'mean'),
        avg_rating=('rating', 'mean'),
        total_products=('title', 'count')
    ).reset_index().sort_values('total_revenue', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(cat_revenue, x='category', y='total_revenue',
                      title='Total Revenue by Category', text='total_products', color='category')
        st.plotly_chart(fig1, use_container_width=True, key="cat_total")
    with col2:
        fig2 = px.scatter(cat_revenue, x='avg_rating', y='avg_revenue',
                          size='total_products', color='category', text='category',
                          title='Rating vs Revenue by Category')
        st.plotly_chart(fig2, use_container_width=True, key="cat_scatter")

    st.info("**Key Takeaway:** Cameras generate the most revenue per product. Laptops win overall because of sheer volume.")

# ============================================================
# PAGE 4 - WHAT DRIVES SALES
# ============================================================
elif page == '4 - What Drives Sales?':
    st.title(" What Drives Sales?")

    st.subheader(" Correlation Matrix")
    features = ['rating', 'number_of_reviews', 'current/discounted_price',
                'discount_pct', 'is_couponed', 'is_free_delivery_enc',
                'is_sponsored_enc', 'is_best_seller_enc']
    corr_matrix = fdf[features + ['bought_in_last_month']].corr()
    fig = px.imshow(corr_matrix, title='What Drives Sales?',
                    color_continuous_scale='RdBu', text_auto='.2f')
    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
    st.info("**Number of reviews** is the most effective factor driving purchases.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Sales by Price Range")
        price_analysis = fdf.groupby('price_range', observed=True)['bought_in_last_month'].mean().reset_index()
        fig = px.bar(price_analysis, x='price_range', y='bought_in_last_month',
                     title='Avg Sales by Price Range', color='bought_in_last_month',
                     color_continuous_scale='Bluered', text='bought_in_last_month')
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True, key="price_range_bar")
        st.info("Low-price products sell the most.")

    with col2:
        st.subheader("Sales by Number of Reviews")
        reviews_analysis = fdf.groupby('reviews_range', observed=True)['bought_in_last_month'].mean().reset_index()
        fig = px.bar(reviews_analysis, x='reviews_range', y='bought_in_last_month',
                     title='Avg Sales by Reviews Range', color='bought_in_last_month',
                     color_continuous_scale='Bluered', text='bought_in_last_month')
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True, key="reviews_range_bar")

    st.markdown("---")

    st.subheader(" Sponsored vs Organic")
    sponsored_analysis = fdf.groupby('is_sponsored', observed=True).agg(
        avg_sales=('bought_in_last_month', 'mean'),
        avg_revenue=('estimated_revenue', 'mean'),
        avg_rating=('rating', 'mean'),
        count=('title', 'count')
    ).reset_index()

    fig3 = make_subplots(rows=1, cols=3, subplot_titles=['Avg Sales', 'Avg Revenue', 'Avg Rating'])
    colors = ['#EF553B', '#00CC96']
    for i, metric in enumerate(['avg_sales', 'avg_revenue', 'avg_rating']):
        fig3.add_trace(go.Bar(
            x=sponsored_analysis['is_sponsored'],
            y=sponsored_analysis[metric],
            marker_color=colors,
            showlegend=False,
            text=sponsored_analysis[metric].round(2),
            textposition='outside'
        ), row=1, col=i+1)
    fig3.update_layout(title='Sponsored vs Organic - Full Comparison', height=400)
    st.plotly_chart(fig3, use_container_width=True, key="sponsored_comparison")
    st.info("**Sponsored** products get more visibility, more sales, and more revenue for Amazon.")

    st.markdown("---")
    st.subheader("Free Delivery Impact")
    delivery_analysis = fdf.groupby('is_free_delivery')['estimated_revenue']\
                           .agg(['mean', 'median', 'count']).reset_index()
    delivery_analysis['is_free_delivery'] = delivery_analysis['is_free_delivery']\
                                            .map({True: 'Free Delivery', False: 'Paid Delivery'})
    fig2 = px.pie(delivery_analysis, names='is_free_delivery', values='mean',
                  title='Average Revenue: Free vs Paid Delivery',
                  color='is_free_delivery',
                  color_discrete_map={'Free Delivery': '#00cc96', 'Paid Delivery': '#EF553B'})
    fig2.update_traces(textposition='inside', textinfo='percent+label+value')
    st.plotly_chart(fig2, use_container_width=True, key="delivery_pie")

# ============================================================
# PAGE 5 - OPPORTUNITIES
# ============================================================
elif page == '5 - Opportunities':
    st.title(" Opportunities")

    st.markdown("""
    ### Strategy
    Products that:
    - Are **priced $20-50** (sweet spot for sales volume)
    - Have **no current discount** (room to add one)
    - Are **not sponsored** yet
    - Are **not Best Sellers** yet (potential to become one)
    - Have **high rating ≥ 4.5**

    Add a **10% discount + sponsor** them → high chance of boosting sales significantly.
    """)

    opportunity = fdf[
        (fdf['current/discounted_price'].between(20, 50)) &
        (fdf['is_couponed'] == 0) &
        (fdf['is_sponsored'] == 'Organic') &
        (fdf['is_best_seller'] != 'Best Seller') &
        (fdf['rating'] >= 4.5)
    ].sort_values('number_of_reviews', ascending=False).head(20)

    st.subheader(f" Top {len(opportunity)} Opportunity Products")
    st.dataframe(
        opportunity[['title', 'current/discounted_price', 'rating',
                      'number_of_reviews', 'bought_in_last_month', 'category']].reset_index(drop=True)
    )

    if len(opportunity) > 0:
        fig = px.scatter(opportunity, x='number_of_reviews', y='rating',
                         size='bought_in_last_month', color='category',
                         hover_data=['title', 'current/discounted_price'],
                         title='Opportunity Products: Reviews vs Rating')
        st.plotly_chart(fig, use_container_width=True, key="opp_scatter")
