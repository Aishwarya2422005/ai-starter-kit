import sys
import streamlit as st
from custom_performance_evaluation import CustomPerformanceEvaluator
from token_benchmark import run_token_benchmark

import streamlit as st


def _initialize_sesion_variables():
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "perf_metrics_history" not in st.session_state:
        st.session_state.perf_metrics_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "chat_disabled" not in st.session_state:
        st.session_state.chat_disabled = True

    # Initialize llm params
    if "do_sample" not in st.session_state:
        st.session_state.do_sample = None
    if "max_tokens_to_generate" not in st.session_state:
        st.session_state.max_tokens_to_generate = None
    if "repetition_penalty" not in st.session_state:
        st.session_state.repetition_penalty = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = None
    if "top_k" not in st.session_state:
        st.session_state.top_k = None
    if "top_p" not in st.session_state:
        st.session_state.top_p = None

def _run_custom_performance_evaluation():

    results_path = "./data/results/llmperf"
    num_concurrent_workers = st.session_state.number_concurrent_workers

    mode = "stream"
    llm_api = "sambastudio"

    run_token_benchmark(
        model=st.session_state.llm,
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
        timeout_s=st.session_state.timeout,
        max_num_completed_requests=st.session_state.number_requests,
        num_concurrent_workers=num_concurrent_workers,
        additional_sampling_params="{}",
        results_dir=results_path,
        user_metadata="",
        llm_api=llm_api,
        mode=mode,
    )

    # evaluator = CustomPerformanceEvaluator(
    #     st.session_state.file_path,
    #     st.session_state.endpoint_url,
    #     st.session_state.project_id,
    #     st.session_state.endpoint_id,
    #     st.session_state.endpoint_api_key,
    #     st.session_state.max_tokens
    # )
    
    # results = evaluator.run_rdu_perf_test()

    # return results

def main():

    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    _initialize_sesion_variables()

    st.title(":orange[SambaNova]Custom Performance Evaluation")
    st.markdown(
            "Here you can select a custom dataset that you want to benchmark performance with. Note that with models that support dynamic \
            batching, you are limited to the number of cpus available on your machine to send concurrent requests."
        )
    
    with st.sidebar:

        ##################
        # File Selection #
        ##################
        st.title("File Selection")
        st.text_input(
            "Full File Path",
            help="", # TODO: Fill in help
            key="file_path"
        )

        #########################
        # Runtime Configuration #
        #########################
        st.title("Runtime Configuration")

        st.slider(
            "Max Output Tokens", 
            min_value=1, 
            max_value=2048, 
            value=256,
            key="max_tokens"
        )

        job_submitted = st.sidebar.button("Run!")

    if job_submitted:

        st.toast("Performance evaluation in progress. This could take a while depending on the dataset size and max tokens setting.")
        with st.spinner("Processing"):
            
            results_df = _run_custom_performance_evaluation()
            


if __name__ == "__main__":
    main()