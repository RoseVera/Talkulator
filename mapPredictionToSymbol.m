function symbol = mapPredictionToSymbol(predictedLabel)
    
    % Add all your 15 trained labels here (Digits 0-9 and Operators)
    
    if ~isstring(predictedLabel)
        predictedLabel = string(predictedLabel);
    end
    
    switch lower(predictedLabel)
        % Digits (Assuming your model predicts the word label)
        case {'zero', '0'}
            symbol = '0';
        case {'one', '1'}
            symbol = '1';
        case {'two', '2'}
            symbol = '2';
        case {'three', '3'}
            symbol = '3';
        case {'four', '4'}
            symbol = '4';
        case {'five', '5'}
            symbol = '5';
        case {'six', '6'}
            symbol = '6';
        case {'seven', '7'}
            symbol = '7';
        case {'eight', '8'}
            symbol = '8';
        case {'nine', '9'}
            
        % Operators
        case {'plus'}
            symbol = '+';
        case {'minus'}
            symbol = '-';
        case {'times'}
            symbol = '*';
        case {'divided'}
            symbol = '/';
        case {'equals'}
            symbol = '=';
            
        % Default/Unknown
        otherwise
            % If the prediction is not explicitly mapped, use the predicted label itself
            symbol = char(predictedLabel);
    end
end