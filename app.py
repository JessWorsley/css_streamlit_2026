import streamlit as st
import pandas as pd
import numpy as np
from streamlit_image_comparison import image_comparison
#import cv2
from PIL import Image
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -------------------------------------------------
# GLOBAL STYLE (bigger fonts + cleaner look)
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 18px;
}
h1, h2, h3 {
    font-weight: 600;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


cmap = ListedColormap([
    "#FF0000",  # invalid (light red)
    "#0000FF"   # valid (light blue)
])

def dynamical_system_generic(tau, parameters):
    x, Omega, K, q = parameters
    dx_dtau = - x * (x - q) + 2 * (x + K + q) - 3 * Omega + 2
    dOmega_dtau = - Omega * (x - 2 * q + 1)
    dK_dtau = 2 * q * K
    dq_dtau = 2 * q**2 + q - K - 1
    return dx_dtau, dOmega_dtau, dK_dtau, dq_dtau

## Constraint on phase space ##
def phase_space_constraint(x, K, q):
    return x / (- K - q - 1) >= 0

## Fixed points (x*, Omega*, K*, q*)
P0 = [-3, -4, 0, 1]
P1 = [0, 0, 0, -1]
P2 = [1, 0, 0, -1]
P3 = [0, 1, 0, 0.5]
P4 = [(5-np.sqrt(73))/4, 0, 0, 0.5]
P5 = [(5+np.sqrt(73))/4, 0, 0, 0.5]
P6 = [0, 0, -1, 0]
P7 = [2, 0, -1, 0]

def make_phase_plot(plane, q=0.5, Omega=1, K=0,
                    density=0.8, show_constraint=True,
                    trajectories=None, figsize=(5,4)):

    fig, ax = plt.subplots(figsize=figsize)

    if plane == "(x, Ω)":
        x = np.linspace(-1, 4, 100)
        Om = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, Om)

        U, V, *_ = dynamical_system_generic(None, [X, Y, K, q])

        if show_constraint:
            valid = phase_space_constraint(X, K, q)
            Z = valid.astype(int)
            ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", alpha=0.25)
            ax.streamplot(X, Y, U, V, density=density)
        else:
            ax.streamplot(X, Y, U, V, density=density)
        
        if q == -1:
            for P in [P1,P2]:
                ax.scatter(P[0], P[1], marker='o', color='r')
                
        if q == 0.5:
            for P in [P3,P4,P5]:
                ax.scatter(P[0], P[1], marker='o', color='r')

        ax.set_title(rf"$q$ = {q}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Omega$")

    elif plane == "(x, q)":
        x = np.linspace(-1, 4, 100)
        qarr = np.linspace(-1, 0.5, 100)
        X, Y = np.meshgrid(x, qarr)

        U = dynamical_system_generic(None, [X, Omega, K, Y])[0]
        V = dynamical_system_generic(None, [X, Omega, K, Y])[3]

        if show_constraint:
            valid = phase_space_constraint(X, K, Y)
            Z = valid.astype(int)
            ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", alpha=0.25)
            ax.streamplot(X, Y, U, V, density=density)
        else:
            ax.streamplot(X, Y, U, V, density=density)

        if Omega == 0:
            for P in [P1,P2,P4,P5]:
                ax.scatter(P[0], P[3], marker='o', color='r')
        if Omega == 1:
            ax.scatter(P3[0], P3[3], marker='o', color='r')

        ax.set_title(rf"Ω = {Omega}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q$")

    # trajectories
    if trajectories:
        for sol in trajectories:
            ax.plot(sol[:,0], sol[:,1], lw=2)

    return fig, ax

# Resize the images
random_img = Image.open('astronomaly/random.png').resize((512,512))
trained_img = Image.open('astronomaly/trained.png').resize((512,512))


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Jess Worsley Research Profile",
    page_icon="📡",
    layout="wide"
)

# -------------------------------------------------
# TOP NAVIGATION TABS  (2A)
# -------------------------------------------------
tab_profile, tab_pubs, tab_projects, tab_contact = st.tabs(
    ["👤 Profile", "📚 Publications", "🚀 Projects", "✉️ Contact"]
)

# =================================================
# 👤 PROFILE TAB
# =================================================
with tab_profile:

    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("Jess Worsley")
        st.header(
            "PhD Candidate in Cosmology  \n"
            "📍 University of Cape Town  |  ✉️ wrsjes002@myuct.ac.za"
        )
    
        st.image("logos.png", width=400)
    
    with col2:
        st.image("profile.jpg", use_container_width=True)


    st.markdown("---")


    # ---------- overview badges
    with st.container(border=True):
        st.markdown("### 🔎 Overview")

        st.markdown("""
🏫 **Institution:** University of Cape Town  
🎓 **Education:** BSc Hons & MSc Astrophysics  
🧠 **Research:** Dark energy • Structure formation • Modified gravity • Machine learning  
💻 **Tools:** Python • Mathematica • CLASS
""")


    st.markdown("")


    # ---------- metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Publications", 3)
    col2.metric("Talks", 8)
    col3.metric("Years of Research", 6)


    st.markdown("")


    # ---------- biography in expander
    with st.expander("📖 Biography", expanded=True):
        st.write("""
        I am a motivated and dependable individual with strong communication, teamwork, and leadership skills. I hold an M.Sc. in Astrophysics with distinction from the University of Cape Town (UCT), and am pursuing a PhD under the supervision of Prof. Dunsby at UCT. My masters research focused on structure formation, perturbation theory, and modified gravity, and my research interests lie in investigating early-universe cosmology and the dark sector using machine learning methods. I have extensive leadership experience, including serving as Chair of UCT ParaSport and an executive member of the Student Sports Union in the transformation portfolio. Additionally, I have over five years of science communication experience, engaging the public through lecturing at the Iziko Planetarium and the Cape Town Science Centre.
        """)


    st.markdown("")


    # ---------- download CV button
    with open("JWorsley_CV_Feb2026.pdf", "rb") as f:
        st.download_button(
            "📄 Download CV",
            f,
            file_name="JWorsley_CV.pdf",
            use_container_width=False
        )


# =================================================
# 📚 PUBLICATIONS TAB
# =================================================
with tab_pubs:

    st.header("📚 Publications")

    publications = pd.read_csv("publications.csv")

    keyword = st.text_input("🔎 Filter by keyword")

    if keyword:
        publications = publications[publications['Keywords'].str.contains(keyword.lower())]

    publications['Title'] = publications.apply(
        lambda x: f'<a href="{x["Link"]}">{x["Title"]}</a>', axis=1
    )

    html = publications[['Title','Year','Authors','Keywords']].to_html(
        escape=False,
        index=False
    )

    st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📈 Publication Trends")
    st.bar_chart(publications["Year"].value_counts().sort_index())


# =================================================
# 🚀 PROJECTS TAB
# =================================================
with tab_projects:

    st.header("🚀 Projects")

    proj_tabs = st.tabs(["Astronomaly", "Dynamical Systems"])


    # ---------- Astronomaly
    with proj_tabs[0]:

        st.set_page_config("Astronomaly")

        st.header("Finding Weird Galaxies with Astronomaly")
        
        st.write("These images show the power of machine learning in identifying anomalous galaxies, by comparing random images from a dataset with those selected by a trained active learning program. For my honours project, I tested Astronomaly -- which had only been tested on optical galaxies up till then -- on radio galaxies. When exploring vast datasets like this, active learning is essential in identifying unique data points. Astronomaly, designed by [Michelle Lochner](https://github.com/MichelleLochner/astronomaly) and her group, utilises active learning to search for particular galaxies in the user's scope, whether that be anomalous shapes or image artifacts.")
        st.write("Can you spot the X-shaped galaxy?")
        
        st.markdown("### Radio Galaxies (GLEAM Database)")
        
        image_comparison(
            img1=random_img,
            img2=trained_img,
            label1="Random",
            label2="Trained",
            make_responsive=False,
        )

    # ---------- Dynamical Systems
    with proj_tabs[1]:

        st.set_page_config("Dynamical Systems")
        st.header("Dynamical Systems in $f(R)$ Gravity")

        st.markdown("When solving the field equations for a chosen $f(R)$ theory, it is beneficial to determine the form of $f(R)$ that yields a desired solution. This is achieved via the various *reconstruction* methods; this is basically a bottom-up approach where the form of $f(R)$ is left entirely general while satisfying specific cosmological conditions. We can formulate a autonomous dynamical system which describes the evolution of parameters relating to the expansion-normalised dynamical dimensionless variables present in $f(R)$ gravity:")

        st.latex(r'''
        \begin{align*}
            \frac{dx}{d\eta} &= -x(x-q) + 2(x+q) - 3\Omega + 2 \\
            \frac{d\Omega}{d\eta} &= -\Omega (x-2q+1) \\
            \frac{dq}{d\eta} &= 2q^2 + q - 1
        \end{align*}
        ''')

        st.markdown(r"where $x=\dot{F}(R)/(HF)$, $\Omega=\rho/(3FH^2)$, and $q$ is the deceleration parameter. For a detailed breakdown, see [MacDevette et al. (2021)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.124040).")

        st.markdown(r"The physical viability of any $f(R)$ theory requires that $F>0$ throughout the physically relevant region of the phase space, and that $F'\ge 0$ at least within the locality of the fixed point corresponding to the matter-dominated epoch. This translates to a constraint for the viable region:")

        st.latex(r'''
        \begin{equation*}
            \frac{x}{-q-1} \ge 0
        \end{equation*}
        ''')

        st.write("Below are streamplots visualising different slices of the dynamical system. The viable regions are shown in blue; the unviable in red. The fixed points of the system are denoted by red dots.")
        
        plane = st.selectbox("Plane", ["(x, Ω)", "(x, q)"])

        if plane == "(x, Ω)":
            q = st.slider("q", -1.5, 1.0, 0.5, 0.1)
            Omega = None
        
        elif plane == "(x, q)":
            Omega = st.slider("Ω", 0.0, 1.0, 0.5, 0.1)
            q = None
            
        density = 1.0
        show_constraint = st.toggle("Show viable region", True)
        
        trajectories = None
        
        fig, _ = make_phase_plot(
            plane,
            q=q,
            Omega=Omega,
            density=density,
            show_constraint=show_constraint,
            trajectories=trajectories,
            figsize=(5,4)
        )

        col1, col2, col3 = st.columns([1, 2, 1]) # Adjust column ratios as needed
        
        with col2:
            st.pyplot(fig, width='content')


# =================================================
# ✉️ CONTACT TAB
# =================================================
with tab_contact:

    st.header("✉️ Contact")

    st.markdown("""
**Email:** wrsjes002@myuct.ac.za  
**GitHub:** https://github.com/JessWorsley  
**LinkedIn:** https://www.linkedin.com/in/jesscjworsley/  
**iNaturalist:** https://www.inaturalist.org/people/jesscjworsley
""")
