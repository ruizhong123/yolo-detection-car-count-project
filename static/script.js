import * as tf from '@tensorflow/tfjs';
import * from "https://cdn.jsdelivr.net/npm/chart.js";



const url = {"% url 'get_data' %"}

const ins_bar_chart_id = document.getElementById('ins_bar_chart_id').getConstext('2d');


const ins_bar_chart = new Chart (
    ins_bar_chart_id, {
        type: 'bar',
        data: {
            label:[],

            // two plots ,red and blue
            datasets:[
                {label: 'Ins Bar Chart',
                    data:[],
                    backgroundColor: 'red'},
                {label: 'Ins Bar Chart',
                    data:[],
                    backgroundColor: 'blue'}
                ]
                },
        options: {scales: {y: {beginAtZero: true}}}});


function updatechart() {

    return fetch(url)
        .then(response => response.json())
        .then(data => {
            
            const modelpredictor = await tf.loadGraphModel('./static/model.json');

            // retain the last 32 data points
            const lag = 32;
            const Last_lagdata = data.slice(-lag);
            
            ins_bar_chart.data.dataset[0].data = modelpredictor.predict(tf.zeros([1,10,1]));
            

            const new_data_len = data.length;

            const old_data = Last_lagdata.slice(-(lag - new_data_len));

            // combine old and new data
            const combined_data = old_data.concat(data);
            normalized_data = combined_data.map((value) => {
                return (value - mean) / std;
            });

            // predict the next 10 data points
            const prediction1 = modelpredictor.predict(normalized_data).data();

            // denormalize the data
            const denormalized = prediction1.map((value) => {
                return value * std + mean;
            });


            //  
            let prediction_list = [];
            prediction_list.push(denormalized);

            // if the prediction list is longer than 10, remove the first element
            
            if (prediction_list.length == 10) {
                prediction_list = [];
                prediction_list.push(denormalized);

            }
             
            ins_bar_chart.data.label = data.datatime;
            ins_bar_chart.data.datasets[0].data = data.polygon1count;
            ins_bar_chart.data.datasets[1].data = denormalized;
            ins_bar_chart.updatechart();

        }

        )
        .catch (error => {
            console.error("Error fetching data:", error);
        }
        );
}




       

async function aquire(url) {
    try {

        // import model from url 
        const modeljson = await aquire('./static/model.json');

        const modelweights = await aquire('./static/group1-shard.bin');

        const modelpredictor = await tf.loadGraphModel(modeljson,  modelweights );

        
        
        const rawdata = [2,3,3,1,3,2,1,3,3,3]
        
        const noral_data= rawdata.map((value) => {

            return (value - mean) / std;
        
        })
        
        let result = modelpredictor.predict(tf.zeros([1,10,1]));

        // Normalize  data , shape [1,10,1]

        const prediction = tf.tensor3d([data],[1,10,1]);
        
        
        const array = prediction.array();

        // 
        const iin_line = new Chart


        
        // denormalize data 
        const denormalized = array.map((value)=> {
            
            return value*std +mean ;
       
        }
    )
        console.log(predict.shape);


         } catch (error) {
                            console.error("Error loading model:", error);
        
                        }
                    }