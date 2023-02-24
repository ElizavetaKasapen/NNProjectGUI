**Project structure:**
- data: includes folder for models weights, server output, data for training (it was not added to the repository because the size is very large)
- models: includes notebooks, which we used to train models in Google Colab, generators' classes which are used for server part
- server.py: user interface
- requirements.txt: requirements to run the project

**Faces Generator** was deployed on the streamlit sharing: https://elizavetakasapen-nnprojectgui-server-wu5j6h.streamlit.app/ <br />
But there is a problem because this server doesn't support CUDA operations. <br />
In this case, it's possible to run user interface using the following command: *streamlit run server.py*<br />

The files for training are in the models/NoteBooks. To run the training code you could use either Google Colab or Jupiter Notebook. Please, make sure that you change the file paths according to your data.


