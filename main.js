
//Model
const model=tf.sequential();

//Hidden Layer Dense-Fully Connected 
//1 Layer so inputShape is required
//units is no of functions/nodes
const hidden = tf.layers.dense(
    {
        units: 4,
        inputShape: [2],
        activation: 'sigmoid'
    }
);

//Output Layer Dense-Fully Connected
//3 outputs
const output = tf.layers.dense(
    {
        units: 1,
        activation: 'sigmoid'
    }
);

//Add the LAyers to the model
model.add(hidden);
model.add(output);

//create a optimizer-> which changes the ratio of weights
//sgd->sequential gradient descent with learning rate of 0.1
const sgdOpt=tf.train.sgd(0.9);

//config of model needs optimizer
//loss function->to calculate error 
//then compile model
model.compile(
        {
            optimizer: sgdOpt,
            loss: tf.losses.meanSquaredError
        }
);

// const xs=tf.tensor2d([
//     [0,1]
// ]);

// let ys=model.predict(xs);
// ys.print();

const xs = tf.tensor2d([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]);

const ys = tf.tensor2d([
    [0],
    [1],
    [1],
    [0],
]);

train().then(()=>{
    console.log('Training Done');
    let yss = model.predict(xs);
    yss.print();
});

async function train(){
    for(let i=0;i<10000;i++){
    const response = await model.fit(xs, ys,{
        shuffle:true,
        epochs:10
    });
    console.log(response.history.loss[0]);
    }
}


