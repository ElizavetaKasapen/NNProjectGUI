Faces Generator was deployed on the streamlit sharing: https://elizavetakasapen-nnprojectgui-server-wu5j6h.streamlit.app/ 
But there is a problem because this server doesn't support CUDA operations. 
In this case, it's possible to run user interface using the following command: *streamlit run server.py*

Project structure:
- data: includes folder for models weights, server output, data for training (it was not added to the repository because the size is very large)
- models: includes notebooks, which we used to train models in Google Colab, generators' classes which are used for server part
- server.py: user interface
- requirements.txt: requirements to run the project
