import time
import open3d as o3d

def main():
    # Crear el visualizador
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    # Aquí puedes agregar geometría al visualizador
    # visualizer.add_geometry(pcd)

    # Iniciar el renderizado en segundo plano, pero no bloquea el código
    visualizer.poll_events()
    visualizer.update_renderer()

    # Tiempo de ejecución del programa
    print("Starting visualization...")
    #visualizer.run()
    time.sleep(2)
    
    visualizer.destroy_window()  # Cerrar el visualizador al finalizar

if __name__ == "__main__":
    start_time = time.time()  # Guardar el tiempo de inicio
    main()
    # Imprimir el tiempo de ejecución después de que se inicie el visualizador
    end_time = time.time()
    print(f"Running time: {end_time - start_time:.2f} seconds")