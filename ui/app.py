from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Coach IA",
    page_icon="J",
    layout="wide",
)


def main() -> None:
    try:
        st.switch_page("pages/2_Coach_IA.py")
    except Exception:
        st.title("Coach IA")
        st.caption("A página principal do Jarvis Trader agora é o Coach IA.")
        st.page_link("pages/2_Coach_IA.py", label="Abrir Coach IA", icon="J")


if __name__ == "__main__":
    main()
