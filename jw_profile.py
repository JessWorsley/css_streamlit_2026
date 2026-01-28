import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

# Set page title
st.set_page_config(page_title="Jess Worsley Resarch Profile", layout="wide")

# Sidebar Menu
st.logo("cosmo_logo.png", icon_image="cosmo_logo.png", link="https://uctcosmology.com/")
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Researcher Profile", "Publications", "Data Explorer", "Contact"],
)

# Dummy STEM data
physics_data = pd.DataFrame({
    "Experiment": ["Alpha Decay", "Beta Decay", "Gamma Ray Analysis", "Quark Study", "Higgs Boson"],
    "Energy (MeV)": [4.2, 1.5, 2.9, 3.4, 7.1],
    "Date": pd.date_range(start="2024-01-01", periods=5),
})

astronomy_data = pd.DataFrame({
    "Celestial Object": ["Mars", "Venus", "Jupiter", "Saturn", "Moon"],
    "Brightness (Magnitude)": [-2.0, -4.6, -1.8, 0.2, -12.7],
    "Observation Date": pd.date_range(start="2024-01-01", periods=5),
})

weather_data = pd.DataFrame({
    "City": ["Cape Town", "London", "New York", "Tokyo", "Sydney"],
    "Temperature (°C)": [25, 10, -3, 15, 30],
    "Humidity (%)": [65, 70, 55, 80, 50],
    "Recorded Date": pd.date_range(start="2024-01-01", periods=5),
})

# Sections based on menu selection
if menu == "Researcher Profile":
    st.title("Welcome! 👋")
    # st.sidebar.header("Profile Options")

    # Collect basic information
    name = "Jess Worsley"
    field = "Astrophysics & Cosmology"
    research = "Dark energy, structure formation, modified gravity, machine learning"
    institution = "University of Cape Town, South Africa"

    # Display basic profile information
    st.header("Overview")
    st.write(f"**Name:**               {name}")
    st.write(f"**Field:**              {field}")
    st.write(f"**Resarch interests:**  {research}")
    st.write(f"**Institution:**        {institution}")

    st.header("Biography")
    st.write("I hold an M.Sc. in Space Science & Astrophysics with distinction from the University of Cape Town (UCT), and am now pursuing a PhD under the supervision of Prof. Dunsby at UCT. My masters research focused on structure formation, perturbation theory, and modified gravity, and my research interests lie in investigating early-universe cosmology and the dark sector using machine learning methods. I have extensive leadership experience, including serving as Chair of UCT ParaSport and an executive member of the Student Sports Union (SSU) in the transformation portfolio. Additionally, I have over five years of science communication experience, engaging the public through lecturing at the Iziko Planetarium and the Cape Town Science Centre.")
    
    st.image(
    "logos.png")

elif menu == "Publications":
    st.title("Publications")
    
    publications = pd.read_csv('publications.csv')
    st.dataframe(publications)

    # Add filtering for year or keyword
    keyword = st.text_input("Filter by keyword", "")
    if keyword:
        filtered = publications[
            publications.apply(lambda row: keyword.lower() in row.astype(str).str.lower().values, axis=1)
        ]
        st.write(f"Filtered Results for '{keyword}':")
        st.dataframe(filtered)
    else:
        st.write("Showing all publications")

    # Publication trends
    if "Year" in publications.columns:
        st.subheader("Publication Trends")
        year_counts = publications["Year"].value_counts().sort_index()
        st.bar_chart(year_counts)
    else:
        st.write("The CSV does not have a 'Year' column to visualize trends.")

elif menu == "Data Explorer":
    st.title("Data Explorer")
    st.sidebar.header("Data Selection")
    
    # Tabbed view for STEM data
    data_option = st.sidebar.selectbox(
        "Choose a dataset to explore", 
        ["Matter Density Data", "Astronomy Observations", "Weather Data"]
    )

    if data_option == "Matter Density Data":
        
        st.set_page_config(layout="wide")

        st.header('Matter Density Evolution in a Flat Universe')
        st.write('Below you can see the evolution of matter density variables with redshift. Ωₘ is the matter density.')
        st.latex(r'''\Omega_m''')
        
        z_arr = np.logspace(-4, 4, 6000) - 1
        
        Om = Planck18.Om(z_arr)
        Or = Planck18.Ogamma(z_arr) + Planck18.Onu(z_arr)
        OL = Planck18.Ode(z_arr)
        Ok = Planck18.Ok(z_arr)
        
        Otot = Om + Or + OL + Ok
        
        log_a = np.log(1 / (1 + z_arr))
        x_logz = -np.log(1 + z_arr)
        
        omegas = {
            r"Ωₘ": Om,
            r"Ωᵣ": Or,
            r"Ωᴧ": OL,
            r"Ωₖ": Ok,
            r"Ωₜ": Otot,
        }

        st.sidebar.header("Controls")
        
        selected = st.sidebar.multiselect(
            "Components",
            list(omegas.keys()),
            default=[r"Ωₘ", r"Ωᵣ", r"Ωᴧ"]
        )
        
        axis_mode = st.sidebar.radio(
            "x-axis",
            ["-ln(1+z)", "redshift z"]
        )
        
        show_unity = st.sidebar.toggle("Show Ω = 1 line", True)
        show_equalities = st.sidebar.toggle("Show equality markers", True)
        
        # -------------------------------------------------
        # Choose x-axis
        # -------------------------------------------------
        if axis_mode == "-ln(1+z)":
            x = x_logz
            xlabel = r"$-\ln(1+z)$"
        
        else:
            x = z_arr
            xlabel = r"$z$"
        
        # -------------------------------------------------
        # Build dataframe dynamically
        # -------------------------------------------------
        data = pd.DataFrame(
            {name: omegas[name] for name in selected},
            index=x
        )
        
        # -------------------------------------------------
        # Equality redshifts
        # -------------------------------------------------
        def find_cross(x, y1, y2):
            i = np.argmin(np.abs(y1 - y2))
            return x[i]
        
        x_eq_mr = find_cross(x, Om, Or)   # matter-radiation
        x_eq_ml = find_cross(x, Om, OL)   # matter-Lambda
        
        # -------------------------------------------------
        # Plot (matplotlib for full control)
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(4, 2))
        
        for col in data.columns:
            ax.plot(data.index, data[col], label=col)
        
        if show_unity:
            ax.axhline(1, linestyle="--", color='k')
        
        if show_equalities:
            ax.axvline(x_eq_mr, linestyle=":", color='k')
            ax.axvline(x_eq_ml, linestyle=":", color='k')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Omega(z)$")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        
        fig.tight_layout()
        
        # -------------------------------------------------
        # Tabs
        # -------------------------------------------------
        tab1, tab2 = st.tabs(["Chart", "Data"])
        
        tab1.pyplot(fig)
        tab2.dataframe(data, use_container_width=True)


    elif data_option == "Astronomy Observations":
        st.write("### Astronomy Observation Data")
        st.dataframe(astronomy_data)
        # Add widget to filter by Brightness
        brightness_filter = st.slider("Filter by Brightness (Magnitude)", -15.0, 5.0, (-15.0, 5.0))
        filtered_astronomy = astronomy_data[
            astronomy_data["Brightness (Magnitude)"].between(brightness_filter[0], brightness_filter[1])
        ]
        st.write(f"Filtered Results for Brightness Range {brightness_filter}:")
        st.dataframe(filtered_astronomy)

    elif data_option == "Weather Data":
        H0 = Planck18.H0
        Om0 = Planck18.Om0
        Or0 = Planck18.Ogamma0 + Planck18.Onu0
        OL0 = Planck18.Ode0
        Ok0 = Planck18.Ok0
        
        def H(z):
            return Planck18.H(z).value
        
        def dOdt(z,O):
            dOm = O[0] * H(z) * ((O[0] - 1) + 2 * O[1] - 2 * O[2])
            dOr = O[1] * H(z) * (O[0] + 2 * (O[1] - 1) - 2 * O[2])
            dOL = O[2] * H(z) * (O[0] + 2 * O[1] - 2 * (O[2] - 1))
            return [dOm, dOr, dOL]
        
        z_arr = np.linspace(-10, 10000, 100000)
        x = -np.log(z_arr + 1)
        y0 = [Om0, Or0, OL0]
        
        fig, ax = plt.subplots(figsize=(7,5))

        ax.plot(x, Planck18.Om(z_arr), label=r'$\Omega_m$', linestyle='--')
        ax.plot(x, Planck18.Ogamma(z_arr) + Planck18.Onu(z_arr), label=r'$\Omega_r$', linestyle='-.')
        ax.plot(x, Planck18.Ode(z_arr), label=r'$\Omega_{\Lambda}$', linestyle=':')
        ax.plot(x, Planck18.Ok(z_arr), label=r'$\Omega_k$', linestyle='--')
        ax.plot(
            x,
            Planck18.Om(z_arr)
            + Planck18.Ogamma(z_arr)
            + Planck18.Onu(z_arr)
            + Planck18.Ode(z_arr)
            + Planck18.Ok(z_arr),
            label=r'$\Omega$',
            linestyle=':'
        )
        
        ax.set_ylabel(r'$\Omega(z)$')
        ax.set_xlabel(r'$-\ln(1 + z)$')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        
        fig.tight_layout()
        st.pyplot(fig)
        



elif menu == "Contact":
    # Add a contact section
    st.header("Contact Information")
    email = "wrsjes002@myuct.ac.za"
    st.write(f"You can reach me at {email}.")