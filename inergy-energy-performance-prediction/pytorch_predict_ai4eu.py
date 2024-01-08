

from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from pytorch_lightning import seed_everything, LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profilers import SimpleProfiler
from torch import nn, optim, rand, sum as tsum, reshape, save
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import joblib
import logging

def calculate_energy_consumption(predictions_unscaled, inputs_unscaled):
    areas_columns = predictions_unscaled[:, :5]
    u_columns = predictions_unscaled[:, 5:10]
    #FOR VISUALIZATION
    #app.all_u_columns = u_columns
   
    #app.all_area_columns = areas_columns[-1]
    
    remaining_columns = predictions_unscaled[:, 10:]
    #app.all_remaining_columns = remaining_columns
    
    # Extracting specific values from the remaining columns
    h = remaining_columns[:, 0]
    specific_heat_gains = remaining_columns[:, 1]

    # Calculate envelope heat losses
    heat_losses = areas_columns * u_columns * (18.9) * 192 * 24 / 1000
    thermal_bridges = np.sum(heat_losses, axis=1, keepdims=True) * 0.03
    envelope_heat_losses = np.sum(np.concatenate((heat_losses, thermal_bridges), axis= 1), axis = 1)
    # thermal
    # Calculate ventilation heat losses
    useful_area = inputs_unscaled[:, 0]
    V = useful_area * 2.5
    ventilation_heat_loss_coefficient = V * h * 0.34
    ventilation_heat_losses = ventilation_heat_loss_coefficient * 18.9 * 192 * 24 / 1000

    # Calculate total heat losses and gains
    total_heat_losses = envelope_heat_losses + ventilation_heat_losses
    total_heat_gains = specific_heat_gains * V

    # Calculate energy consumption
    ratio = np.abs(total_heat_gains / total_heat_losses)
    if inputs_unscaled[-1][-1] == 0:
        # Heavy buildings
        building_type = 54.2
    else:
        # Light buildings
        building_type = 23.1
    building_time_constant = building_type * useful_area / (total_heat_losses / (192 * 24) / 18.9 * 10 ** 6)
    divide = building_time_constant / 30
    numerical_parameter = divide + 0.8
    heat_gain_usage_factor = (1 - ratio ** numerical_parameter) / (1 - ratio ** (numerical_parameter + 1))
    energy_consumption = total_heat_losses - total_heat_gains * heat_gain_usage_factor
    
    return energy_consumption / 1000

def scale(data):
    scaler = joblib.load('/models/my_scaler_input_pytorch.save')
    scaled_data = scaler.transform(data.reshape(1,-1))
    return scaled_data

def unscale_output(predicted_data):
    scaler = joblib.load('/models/my_scaler_output_pytorch.save')
    unscaled_output_data = scaler.inverse_transform(predicted_data)
    return unscaled_output_data

def unscale_input(input_data):
    scaler = joblib.load('/models/my_scaler_input.save')
    unscaled_input_data = scaler.inverse_transform(input_data.reshape(1,-1))
    return unscaled_input_data

def predict(params):
    total_area = float(params["total_area"])
    useful_area = int(params["useful_area"])
    apartments = int(params["apartments"])
    floors = int(params["floors"])
    serie = int(params["serie"])
    Heavy = int(params["Heavy"])
    Light = int(params["Light"])
    #print('vasilis')
    encoded_serie = encode(serie)
    encoded_serie = encoded_serie[0]
    encoded_serie_columns = [int(i == encoded_serie) for i in range(12)]
  # Create a numpy ndarray from the values and reshape it
    input = np.array([useful_area, floors, apartments, total_area] + encoded_serie_columns + [Heavy ,Light]).reshape(1, -1)
    data_scaled = scale(input)
    model = DNN.load_from_checkpoint('/checkpoint_200/epoch=12-step=2548.ckpt')
    tensor_input = torch.from_numpy(data_scaled.copy())
    tensor_input = tensor_input.to(torch.float32)
    y_pred = model(tensor_input)
    y_pred = y_pred.to(torch.float64)
    input_unscaled = unscale_input(data_scaled)
    output_unscaled = unscale_output(y_pred.detach().numpy())
    energy_consumption = calculate_energy_consumption(output_unscaled, input_unscaled)
    return energy_consumption

def encode(serie):
    try:
        label_encoder = joblib.load('/models/my_encoder.save')

        str_serie_value = str(serie)
        for i in label_encoder.classes_:
            if i == str_serie_value:
                encoded_serie = label_encoder.transform([str_serie_value])



    except Exception as e:
        return {'Something went wrong with the encoder': str(e)}

    #encoded_serie = label_encoder.transform([str_serie_value])[0]
    return(encoded_serie)

class DNN(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, scaler_input,scaler_output,scaler_validation):
        super(DNN, self).__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        self.scaler_validation = scaler_validation
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        #x = torch.relu(x)
        #x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc3(x)
        #print(x.shape)
        #x = x.unsqueeze(-1)
        #x = self.calculate_energy_consumption(x,self.inputs)
        
        #x= self.calculate_energy_consumption()
        #x = self.scaler.inverse(x)
        #prepei na gurnaei kai to y(energy consumption)
        #mallon na gurnaei kai ta heat losses gia to pie chart
        return x
    
    #TRAINING
    def training_step(self, batch, batch_idx):
        inputs, targets, val_energy_consumption = batch
       
        #self.inputs = inputs
        y_hat = self.forward(inputs)
        loss = self.custom_loss(y_hat, targets, inputs, val_energy_consumption)  
        #print(y_hat)
        #self.log( 'predictions' , y_hat)
        self.log('train_loss', loss.mean(), on_epoch=True)
        
      
        return loss.mean()
    
    #VALIDATION
    def validation_step(self, batch, batch_idx):
        inputs, targets, val_energy_consumption = batch
        
        #self.inputs = inputs
        y_hat = self.forward(inputs)
        loss = self.custom_loss(y_hat, targets, inputs, val_energy_consumption)

        mse = torch.mean(torch.pow(torch.abs(y_hat - targets),2))
        self.log("val_loss", mse)
        self.log("avg_val_loss", loss.mean(), on_epoch=True)  
        return {'val_loss': mse}
        
    #TESTING
    def test_step(self, batch, batch_idx):
        inputs, targets, val_energy_consumption = batch
        
        y_hat = self.forward(inputs)
        loss = self.custom_loss(y_hat, targets, inputs, val_energy_consumption)

        self.log('test_loss', loss.mean(), on_epoch=True)

        mse = torch.mean(torch.pow(torch.abs(y_hat - targets),2))
        self.log('mse', mse, on_epoch=True)
        return {"test_loss": loss, "mse": mse}

    #PREDICTION
    def predict_step(self, batch, batch_idx):
        inputs, targets, val_energy_consumption = batch
       
        predictions = self.forward(inputs)
        energy_consumption = self.calculate_energy_consumption(predictions,inputs)
        """
        predictions_numpy = predictions.detach().numpy()
        subset1 = predictions_numpy[:, :6]  
        subset2 = predictions_numpy[:, 6:]
        unscaled_subset1 = self.scaler_output_test_areas.inverse_transform(subset1)
        unscaled_subset2 = self.scaler_output_test_U.inverse_transform(subset2)
        predictions_unscaled = np.concatenate((unscaled_subset1, unscaled_subset2), axis=1)
        """
        #predictions_unscaled = self.scaler_output.inverse_transform(predictions.detach().numpy())
        #predictions_unscaled = torch.from_numpy(predictions_unscaled)
        predictions_unscaled = self.scaler_output.inverse_transform(predictions.detach().numpy())
        predictions_unscaled = torch.from_numpy(predictions_unscaled.copy())

        return energy_consumption, predictions_unscaled
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }
        #return optimizer
    
    def custom_loss(self, y_hat, targets, inputs, val_energy_consumption):
        
        #print(y_hat.shape)
        #print(targets.shape)
       
        errors = torch.abs(y_hat - targets)
        mean_errors = torch.mean(torch.pow(errors,2), dim=1)
        
        
        
        a = torch.where(val_energy_consumption != 0, torch.tensor(1.0), torch.tensor(0.0))

     
        
        
        energy_consumption = self.calculate_energy_consumption(y_hat, inputs)
        #print(energy_consumption)
        #print(inputs)
        # o mpampas sotiris eipe na vgalw ton scaler eksw gia na mh ginetai scale ana batch, auto
      
        #val_energy_consumption = self.scaler_validation.fit_transform(val_energy_consumption)
       
        energy_consumption = self.scaler_validation.transform(energy_consumption.reshape(-1,1))
        #print(energy_consumption)
        #new min max scaler
        #mean squared error
        #print(a)
        
        energy_consumption_absolute_loss = torch.abs(val_energy_consumption-energy_consumption)
        energy_consumption_mean_absolute_loss = torch.mean(torch.pow(energy_consumption_absolute_loss,2),dim=1)
        #print(energy_consumption_mean_absolute_loss)
        # Calculate the energy consumption using predictions F(x)
        #energy_consumption = self.calculate_energy_consumption(y_hat, inputs)
        #print(energy_consumption)
       
        a_expanded = a.view(-1)
        
        energy_consumption_mean_absolute_square_loss = a_expanded * energy_consumption_mean_absolute_loss
        loss = mean_errors + energy_consumption_mean_absolute_square_loss
        #loss = torch.cat((mean_errors.unsqueeze(1), energy_consumption_mean_absolute_square_loss.unsqueeze(1)), dim=1)
        #print(loss)
        
        return loss
    
    def calculate_energy_consumption(self, predictions, inputs):
        
        #print(prediction
        # s)
        #print(inputs)
        """
        predictions_numpy = predictions.detach().numpy()
        subset1 = predictions_numpy[:, :6]  
        subset2 = predictions_numpy[:, 6:]
        unscaled_subset1 = self.scaler_output_areas.inverse_transform(subset1)
        unscaled_subset2 = self.scaler_output_U.inverse_transform(subset2)
        predictions_unscaled = np.concatenate((unscaled_subset1, unscaled_subset2), axis=1)

        
        """
        predictions_unscaled = self.scaler_output.inverse_transform(predictions.detach().numpy())
        predictions_unscaled = torch.from_numpy(predictions_unscaled)
        inputs_unscaled = self.scaler_input.inverse_transform(inputs.detach().numpy())
        inputs_unscaled = torch.from_numpy(inputs_unscaled)
        
       
        areas_columns = predictions_unscaled[:, :5]
        #print(predictions_unscaled)
        u_columns = predictions_unscaled[:, 5:10]
        #remaining_columns = [:,-2] 
        remaining_columns = predictions_unscaled[:, 10:]

# Extracting specific values from the remaining columns
        h = remaining_columns[:, 0]
        specific_heat_gains = remaining_columns[:, 1]
        #specific_heat_gains = predictions_unscaled[:, 5]#specific heat gains
          
        #h = predictions_unscaled[:,-1] #air exchange rate
      
        #print(areas_columns)
        #print(u_columns)
        #print(remaining_columns.size())
        #envelope heat losses
        heat_losses = areas_columns * u_columns * (18.9) * 192 * 24 / 1000
      
        thermal_bridges = torch.sum(heat_losses, dim=1, keepdim=True) * 0.03
        
        envelope_heat_losses = torch.sum(torch.cat((heat_losses, thermal_bridges), dim=1), dim=1)
        #envelope_heat_losses = torch.add(heat_losses,thermal_bridges)
        

        #ventilation heat losses
        useful_area = inputs_unscaled[:,0]
        
        V = useful_area * 2.5 # 2.8 is average indoor height
        
        
        ventilation_heat_loss_coefficient = V * h *0.34
        ventilation_heat_losses = ventilation_heat_loss_coefficient * 18.9 * 192 * 24 / 1000
        
        
        #total heat losses
        total_heat_losses = torch.add(envelope_heat_losses,ventilation_heat_losses)
       
        total_heat_gains = specific_heat_gains*V

        #print(total_heat_losses)
        #final calculations tsekare gia diairesh me to 0
        ratio = torch.abs(total_heat_gains/total_heat_losses)
        if inputs_unscaled[-1][-1] == 0:
            # heavy buildings
            building_type = 54.2 #inputs['building_type'] IT SHOULD BE IN INPUTS | currently the value is for heavy tsekare gia diairesh me to 0
        else:
            # light buildings
            building_type = 23.1
        building_time_constant = building_type * useful_area / ( total_heat_losses / (192*24) / 18.9 * 10 ** 6)
        divide = torch.div(building_time_constant,30)
        numerical_parameter = torch.add(divide,0.8)
        
        heat_gain_usage_factor = torch.div((1 - torch.pow(ratio,numerical_parameter)) , (1 - torch.pow(ratio, torch.add(numerical_parameter,1))))
        
        energy_consumption = torch.sub(total_heat_losses, torch.mul(total_heat_gains,heat_gain_usage_factor))
        


        #print(energy_consumption/1000)

        
        #energy_consumption = torch.sum(predictions)
        
        return energy_consumption/1000
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1, shuffle=False)
    
    def predict_dataloader(self):
        return self.test_dataloader()
    

